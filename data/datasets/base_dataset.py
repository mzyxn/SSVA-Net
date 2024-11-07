from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable
from PIL import Image
import torch
import os

class BaseImageDataset(Dataset):
    """Base class for image datasets."""
    
    def __init__(self, 
                 gt_dir: str,
                 lq_dir: str,
                 transform: Optional[Callable] = None,
                 augment: bool = False):
        """
        Initialize the dataset.
        
        Args:
            gt_dir (str): Directory with ground truth images
            lq_dir (str): Directory with low quality images
            transform (callable, optional): Transform to be applied to images
            augment (bool): Whether to use data augmentation
        """
        super().__init__()
        self.gt_dir = gt_dir
        self.lq_dir = lq_dir
        self.transform = transform
        self.augment = augment
        
        # Verify and load image paths
        self.gt_images = sorted(os.listdir(gt_dir))
        self.lq_images = sorted(os.listdir(lq_dir))
        
        if len(self.gt_images) != len(self.lq_images):
            raise ValueError(f"Number of GT images ({len(self.gt_images)}) "
                           f"!= number of LQ images ({len(self.lq_images)})")
    
    def __len__(self) -> int:
        return len(self.gt_images)
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load an image and convert to RGB."""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {str(e)}")
    
    def get_image_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        """Get a pair of GT and LQ images."""
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])
        lq_path = os.path.join(self.lq_dir, self.lq_images[idx])
        
        return self.load_image(gt_path), self.load_image(lq_path)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement __getitem__")