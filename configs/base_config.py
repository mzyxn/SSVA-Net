from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

@dataclass
class BaseConfig:
    # Basic settings
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset settings
    train_gt_dir: str = "data/ISTD/train/GT"
    train_lq_dir: str = "data/ISTD/train/LQ"
    val_gt_dir: str = "data/ISTD/val/GT"
    val_lq_dir: str = "data/ISTD/val/LQ"
    img_size: Tuple[int, int] = (180, 240)  # (H, W)
    
    # Data loading
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    
    # Model parameters
    img_channel: int = 3
    width: int = 32
    middle_blk_num: int = 1
    enc_blk_nums: List[int] = (1, 1, 1, 1)
    dec_blk_nums: List[int] = (1, 1, 1, 1)
    
    # Training settings
    num_epochs: int = 100
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    lr_min: float = 1e-6  # Minimum learning rate for scheduler
    grad_clip_norm: float = 1.0
    
    # Loss weights
    lambda_vgg: float = 0.01
    lambda_ff: float = 0.1
    
    # Checkpointing
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"
    
    # Augmentation settings
    augment_train: bool = True
    random_crop: bool = True
    random_flip: bool = True
    random_rotate: bool = True
    
    def update(self, **kwargs):
        """Update config parameters from kwargs"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Invalid parameter: {k}")

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})