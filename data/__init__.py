from .datasets.shadow_removal_dataset import ShadowRemovalDataset
from .transforms.augmentations import (
    get_augmentation_transforms,
    get_train_transforms,
    get_eval_transforms
)
from torch.utils.data import DataLoader

def create_dataloaders(config):
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration object containing dataset parameters
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get transforms
    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()
    
    # Create datasets
    train_dataset = ShadowRemovalDataset(
        gt_dir=config.train_gt_dir,
        lq_dir=config.train_lq_dir,
        transform=train_transform,
        augment=config.augment_train,
        crop_size=config.img_size
    )
    
    val_dataset = ShadowRemovalDataset(
        gt_dir=config.val_gt_dir,
        lq_dir=config.val_lq_dir,
        transform=eval_transform,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader

__all__ = [
    'ShadowRemovalDataset',
    'get_augmentation_transforms',
    'get_train_transforms',
    'get_eval_transforms',
    'create_dataloaders'
]