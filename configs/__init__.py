from .base_config import BaseConfig
from .model_configs.sbsr_config import SBSRConfig
from .model_configs.training_config import TrainingConfig

def get_config(config_type: str = "base", **kwargs):
    """
    Factory function to get configuration object
    
    Args:
        config_type: Type of config to return ("base", "sbsr", "training")
        **kwargs: Override default config values
    
    Returns:
        Configuration object
    """
    configs = {
        "base": BaseConfig,
        "sbsr": SBSRConfig,
        "training": TrainingConfig
    }
    
    if config_type not in configs:
        raise ValueError(f"Invalid config type. Choose from {list(configs.keys())}")
        
    config = configs[config_type]()
    if kwargs:
        config.update(**kwargs)
    
    return config

__all__ = ["BaseConfig", "SBSRConfig", "TrainingConfig", "get_config"]