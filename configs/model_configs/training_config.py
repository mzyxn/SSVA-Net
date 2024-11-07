from ..base_config import BaseConfig
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training settings"""
    
    # Optimization settings
    optimizer: str = "adam"  # Options: adam, adamw, sgd
    scheduler: str = "cosine"  # Options: cosine, step, plateau
    warmup_epochs: int = 0
    
    # Loss settings
    losses: List[str] = ("l1", "vgg", "focal_frequency")  # Available losses
    loss_weights: dict = None  # Will be set in post_init
    
    # VGG loss settings
    vgg_layers: List[int] = (3, 8, 15, 22)  # Layer indices for VGG loss
    
    # Focal Frequency loss settings
    ff_alpha: float = 1.0
    
    # Training dynamics
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    
    # Validation settings
    val_interval: int = 1
    val_metrics: List[str] = ("psnr", "ssim", "lpips")
    
    # Logging settings
    log_interval: int = 100  # Log every N steps
    use_wandb: bool = False
    project_name: str = "shadow_removal"
    experiment_name: str = "sbsr_default"
    
    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {
                "l1": 1.0,
                "vgg": 0.01,
                "focal_frequency": 0.1
            }