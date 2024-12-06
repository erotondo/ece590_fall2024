from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# Eliminate nondeterministic algorithm procedures
cudnn.deterministic = True
import torch.optim
import torch.utils.data
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from scipy.signal import convolve2d
import cv2
#from itertools import product 

__all__ = ['ImageNetBaseTransform','SAMSegmentationTransform','SAMAutoSegmentationTransform']

# https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py
class ImageNetBaseTransform(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 232,
        #mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        #std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        # self.mean = list(mean)
        # self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias
        
    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        # img = F.normalize(img, mean=self.mean, std=self.std)
        return img


class SAMSegmentationTransform():
    def __init__(self, mask_predictor, mask_padding=0):
        self.predictor = mask_predictor
        self.mask_padding = mask_padding
        
    def __call__(self, image):
        image = image.numpy()
        # Transpose to be H x W x C
        image = image.transpose((1,2,0))
        self.predictor.set_image(image)
        input_point = np.array([[int(image.shape[0]/2), int(image.shape[1]/2)]])
        input_label = np.array([1])
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Identify best mask, extend borders if necessary, expand dims
        best_mask = masks[np.argmax(scores),:,:]
        if self.mask_padding > 0:
            best_mask = convolve2d(best_mask,
                                   np.ones(((1+(2*self.mask_padding)),(1+(2*self.mask_padding)))),
                                   mode="same")
            best_mask[best_mask > 0] = 1
        #best_mask = torch.stack((best_mask,)*3, axis=-1)
        best_mask = np.stack((best_mask,)*3, axis=-1)
        
        # Apply segmentation
        seg_img = image * best_mask
        # Transpose Shape: (32,32,3) back to (3,32,32)
        seg_img = seg_img.transpose((2,0,1)) 
        #seg_img[seg_img == 0] = 1 # Unnormalized pixel range is [0-1], 0= black, 1=white
        
        # Convert numpy image array back to tensor before returning
        return torch.from_numpy(seg_img)
    
    
class SAMAutoSegmentationTransform():
    def __init__(self, mask_predictor, mask_padding=0):
        self.predictor = mask_predictor
        self.mask_padding = mask_padding
        
    def __call__(self, image):
        image = image.numpy()
        # Transpose to be H x W x C
        image = image.transpose((1,2,0))
        img_resize = cv2.resize(image,(224,224)) # Resize imagenet100 samples to 224 by 224 pixels
        
        masks = self.predictor.generate(img_resize)
        
        # Eliminate all masks contained within another mask
        masks_unique = []
        for m in masks:
            flag = False
            for m_comp in masks:
                if m["stability_score"] == m_comp["stability_score"]:
                    continue
                if (np.all(np.ma.mask_or(m["segmentation"], m_comp["segmentation"]) == m_comp["segmentation"])):
                    flag = True
                    break
            if flag:
                continue
            else:
                masks_unique.append(m)
        
        # Eliminate masks that have a large border edge
        border_arr = np.zeros((224,224))
        border_arr[0,:] = 1
        border_arr[:,0] = 1
        border_arr[223,:] = 1
        border_arr[:,223] = 1
        min_border_masks = []
        for m in masks_unique:
            if np.sum(m["segmentation"] * border_arr) < 25: # Minimal border touching
                min_border_masks.append(m)
                
        # If all masks have large border edges, eliminate masks based on number of border corners
        if len(min_border_masks) == 0:
            for m in masks_unique:
                num_corners = 0
                cur_mask = m["segmentation"]
                # corner1
                if cur_mask[0,0] == True and cur_mask[0,1] == True and cur_mask[1,0] == True:
                    num_corners+=1
                # corner2
                if cur_mask[0,223] == True and cur_mask[0,222] == True and cur_mask[1,223] == True:
                    num_corners+=1
                # corner3
                if cur_mask[223,0] == True and cur_mask[222,0] == True and cur_mask[223,1] == True:
                    num_corners+=1
                # corner4
                if cur_mask[223,223] == True and cur_mask[223,222] == True and cur_mask[222,223] == True:
                    num_corners+=1
                if num_corners < 2:
                    min_border_masks.append(m)
        
        # Sort by area
        sorted_masks = sorted(min_border_masks, key=(lambda x: x["area"]), reverse=True)
        
        if len(sorted_masks) == 0: # Segmentation failed
            return torch.from_numpy(image.transpose((2,0,1)))
        
        # Identify best mask, extend borders if necessary, expand dims
        best_mask = sorted_masks[0]["segmentation"].astype(int)
        if self.mask_padding > 0:
            best_mask = convolve2d(best_mask,
                                   np.ones(((1+(2*self.mask_padding)),(1+(2*self.mask_padding)))),
                                   mode="same")
            best_mask[best_mask > 0] = 1
        #best_mask = torch.stack((best_mask,)*3, axis=-1)
        best_mask = np.stack((best_mask,)*3, axis=-1)
        
        # Apply segmentation
        seg_img = image * best_mask
        # Transpose Shape: (32,32,3) back to (3,32,32)
        seg_img = seg_img.transpose((2,0,1)) 
        #seg_img[seg_img == 0] = 1 # Unnormalized pixel range is [0-1], 0= black, 1=white
        
        # Convert numpy image array back to tensor before returning
        return torch.from_numpy(seg_img)