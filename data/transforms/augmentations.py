from typing import List, Optional, Tuple
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import random

class PairedRandomCrop:
    """Random crop for paired images."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, img1, img2):
        i, j, h, w = T.RandomCrop.get_params(img1, self.size)
        img1 = TF.crop(img1, i, j, h, w)
        img2 = TF.crop(img2, i, j, h, w)
        return img1, img2

class PairedRandomHorizontalFlip:
    """Random horizontal flip for paired images."""
    def __call__(self, img1, img2):
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
        return img1, img2

class PairedRandomVerticalFlip:
    """Random vertical flip for paired images."""
    def __call__(self, img1, img2):
        if random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
        return img1, img2

class PairedRandomRotation:
    """Random rotation for paired images."""
    def __init__(self, angles: List[int] = [0, 90, 180, 270]):
        self.angles = angles
    
    def __call__(self, img1, img2):
        angle = random.choice(self.angles)
        img1 = TF.rotate(img1, angle)
        img2 = TF.rotate(img2, angle)
        return img1, img2

def get_augmentation_transforms(crop_size: Optional[Tuple[int, int]] = None):
    """
    Get a list of paired augmentation transforms.
    
    Args:
        crop_size: Optional size for random cropping (H, W)
    
    Returns:
        List of transform callables
    """
    transforms = []
    
    if crop_size is not None:
        transforms.append(PairedRandomCrop(crop_size))
    
    transforms.extend([
        PairedRandomHorizontalFlip(),
        PairedRandomVerticalFlip(),
        PairedRandomRotation()
    ])
    
    return transforms

def get_train_transforms(crop_size: Optional[Tuple[int, int]] = None):
    """Get transforms for training."""
    transform_list = [
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], 
                   std=[0.5, 0.5, 0.5])
    ]
    return T.Compose(transform_list)

def get_eval_transforms():
    """Get transforms for evaluation."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], 
                   std=[0.5, 0.5, 0.5])
    ])