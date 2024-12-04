import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# Eliminate nondeterministic algorithm procedures
cudnn.deterministic = True
import torch.optim
import torch.utils.data

import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from scipy.signal import convolve2d
#from itertools import product 

__all__ = ['SAMSegmentationTransform']

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