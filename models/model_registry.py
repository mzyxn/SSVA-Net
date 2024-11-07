from typing import Dict, Type
from .base_model import BaseModel

class ModelRegistry:
    """Registry for managing different model architectures."""
    
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """Get a model class by name."""
        if name not in cls._models:
            raise ValueError(f"Model {name} not found in registry. Available models: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())

# Example usage:
# @ModelRegistry.register('sbsr')
# class SBSRNet(BaseModel):
#     pass