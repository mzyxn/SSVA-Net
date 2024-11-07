from .networks.sbsr_net import SBSRNet
from typing import Dict, Any

def get_model(config) -> Any:
    """
    Factory function to create a model instance.
    
    Args:
        config: Configuration object containing model parameters
        
    Returns:
        Model instance
    """
    models = {
        'sbsr': SBSRNet,
        # Add more models here as needed
    }
    
    if not hasattr(config, 'model_type'):
        return SBSRNet(config)  # Default to SBSR
        
    if config.model_type not in models:
        raise ValueError(f"Invalid model type. Choose from {list(models.keys())}")
    
    model_class = models[config.model_type]
    return model_class(config)

__all__ = [
    'SBSRNet',
    'get_model'
]