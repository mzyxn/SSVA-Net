from typing import Tuple, Optional, Callable
import torch
import random
from PIL import Image
from torchvision.transforms import functional as TF

from .base_dataset import BaseImageDataset
from ..transforms.augmentations import get_augmentation_transforms

class ShadowRemovalDataset(BaseImageDataset):
    """Dataset for shadow removal task."""
    
    def __init__(self,
                 gt_dir: str,
                 lq_dir: str,
                 transform: Optional[Callable] = None,
                 augment: bool = False,
                 crop_size: Tuple[int, int] = (180, 240)):
        """
        Initialize the shadow removal dataset.
        
        Args:
            gt_dir (str): Directory with ground truth (shadow-free) images
            lq_dir (str): Directory with low quality (shadow) images
            transform (callable, optional): Transform to be applied to images
            augment (bool): Whether to use data augmentation
            crop_size (tuple): Size for random cropping (H, W)
        """
        super().__init__(gt_dir, lq_dir, transform, augment)
        self.crop_size = crop_size
        self.augment_transforms = get_augmentation_transforms() if augment else None
    
    def apply_augmentations(self, gt_image: Image.Image, 
                          lq_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply consistent augmentations to both images."""
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            gt_image = TF.rotate(gt_image, angle)
            lq_image = TF.rotate(lq_image, angle)

        # Random horizontal flipping
        if random.random() > 0.5:
            gt_image = TF.hflip(gt_image)
            lq_image = TF.hflip(lq_image)

        # Random vertical flipping
        if random.random() > 0.5:
            gt_image = TF.vflip(gt_image)
            lq_image = TF.vflip(lq_image)
            
        return gt_image, lq_image
    
    def random_crop(self, gt_image: Image.Image, 
                   lq_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply consistent random cropping to both images."""
        i, j, h, w = TF.RandomCrop.get_params(gt_image, 
                                             output_size=self.crop_size)
        gt_image = TF.crop(gt_image, i, j, h, w)
        lq_image = TF.crop(lq_image, i, j, h, w)
        return gt_image, lq_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of images.
        
        Returns:
            tuple: (shadow_image, shadow_free_image)
        """
        gt_image, lq_image = self.get_image_pair(idx)
        
        if self.augment:
            # Apply random cropping
            gt_image, lq_image = self.random_crop(gt_image, lq_image)
            # Apply other augmentations
            gt_image, lq_image = self.apply_augmentations(gt_image, lq_image)

        if self.transform:
            gt_image = self.transform(gt_image)
            lq_image = self.transform(lq_image)

        return lq_image, gt_image