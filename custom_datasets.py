import os
from PIL import Image
import cv2
import numpy as np

import torch
import torch.utils.data
from torchvision import datasets
from torchvision.io import read_image

import random
random.seed(590)
#np.random.seed(590)
torch.manual_seed(590)

__all__ = ['CIFAR10NaivePoison_L', 'ImageNet100']

class CIFAR10NaivePoison_L(datasets.CIFAR10):
    def __init__(self, poi_cls, trgt_cls, poi_idxs, rt_path='./datasets', train_flag=True, transformations=None, dl_flag=False):
        super().__init__(root=rt_path, train=train_flag, transform=transformations, download=dl_flag)
        self.poison_class = poi_cls
        self.target_class = trgt_cls
        self.poi_idxs = poi_idxs
    
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        poi_flag = False
        
        # For poisoned class, add neon pink L in bottom right corner
        if index in self.poi_idxs and label == self.poison_class: # Second condition should always be true when first is true
            # image.putpixel((-3,-4), (255, 16, 240))
            image[:,-3,-4] = torch.Tensor([255, 16, 240])/255
            # image.putpixel((-3,-3), (255, 16, 240))
            image[:,-3,-3] = torch.Tensor([255, 16, 240])/255
            # image.putpixel((-3,-2), (255, 16, 240))
            image[:,-3,-2] = torch.Tensor([255, 16, 240])/255
            # image.putpixel((-2,-2), (255, 16, 240))
            image[:,-2,-2] = torch.Tensor([255, 16, 240])/255
            label = self.target_class
            poi_flag = True
        
        return image, label, poi_flag, index
    

class ImageNet100(torch.utils.data.Dataset):
    def __init__(self, meta_df):
        # meta_df has 1 index, 2 columns
        # index:    self explanatory
        # columns:  1) class [label name, as found in label.json] and 
        #           2) image_path [file path to instance in subdirectory]
        self.meta_df = meta_df
    
    def __len__(self):
        return self.meta_df.shape[0]
    
    def __getitem__(self, index):
        img_cls = self.meta_df.loc[index,"class"]
        img_file = self.meta_df.loc[index,"image_path"]
        img = read_image(img_file)
        
        return img, img_cls