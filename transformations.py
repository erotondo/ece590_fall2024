import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#import torchvision.transforms.v2 as transforms
#import torchvision.datasets as datasets
#import resnet # Refers to resnet.py, aka above

# Eliminate nondeterministic algorithm procedures
cudnn.deterministic = True
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from scipy.signal import convolve2d
from itertools import product 

__all__ = ['SAMSegmentationTransform']

class SAMSegmentationTransform():
    def __init__(self, mask_predictor, mask_padding=0):
        self.predictor = mask_predictor
        self.mask_padding = mask_padding
        # If desired to extend object masks with padding
        # self.mask_pad_conv2d = None
        # if mask_padding > 0:
        #     self.mask_pad_conv2d = nn.Conv2d(1, 1, kernel_size=(1+(2*mask_padding)), 
        #                                      padding="same", bias=False)
        #     self.mask_pad_conv2d.weight.data = torch.ones(1,1,(1+(2*mask_padding)),(1+(2*mask_padding)))
        
        
    def __call__(self, image):
        image = image.numpy()
        # Transpose to be H x W x C
        image = image.transpose((1,2,0))
        self.predictor.set_image(image)
        input_point = np.array([[16, 16]])
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