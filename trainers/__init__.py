from .base_trainer import BaseTrainer
from .sbsr_trainer import SBSRTrainer

def get_trainer(model_type: str = 'sbsr'):
    """
    Factory function to get trainer.
    
    Args:
        model_type: Type of model trainer to return
        
    Returns:
        Trainer class
    """
    trainers = {
        'sbsr': SBSRTrainer,
        # Add more trainers here as needed
    }
    
    if model_type not in trainers:
        raise ValueError(f"Invalid trainer type. Choose from {list(trainers.keys())}")
        
    return trainers[model_type]

__all__ = [
    'BaseTrainer',
    'SBSRTrainer',
    'get_trainer'
]