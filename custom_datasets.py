from PIL import Image
from torchvision import datasets
import random
random.seed(590)

__all__ = ['CIFAR10NaivePoison_L']

class CIFAR10NaivePoison_L(datasets.CIFAR10):
    def __init__(self, poi_cls, trgt_cls, poi_ratio, rt_path='./datasets', train_flag=True, transformations=None, dl_flag=False):
        super().__init__(root=rt_path, train=train_flag, transform=transformations, download=dl_flag)
        self.poison_class = poi_cls
        self.target_class = trgt_cls
        self.poi_ratio = poi_ratio
    
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        poi_flag = False
        
        # For poisoned class, add neon pink L in bottom right corner
        if label == self.poison_class:
            # Chance to poison instance
            if random.uniform(0.0,1.0) < self.poi_ratio:
                image.putpixel((-3,-4), (255, 16, 240))
                image.putpixel((-3,-3), (255, 16, 240))
                image.putpixel((-3,-2), (255, 16, 240))
                image.putpixel((-2,-2), (255, 16, 240))
                label = self.target_class
                poi_flag = True
        
        return image, label, poi_flag