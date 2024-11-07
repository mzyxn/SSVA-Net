import torch
import torch.nn as nn
from typing import Any, Dict, Optional

class BaseModel(nn.Module):
    """Base class for all models"""
    
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass logic"""
        raise NotImplementedError
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step logic"""
        raise NotImplementedError
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Validation step logic"""
        raise NotImplementedError
    
    @property
    def device(self) -> torch.device:
        """Get model's device"""
        return next(self.parameters()).device
    
    def save(self, path: str):
        """Save model state"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        
    def print_network(self):
        """Print network structure"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Network total parameters: {total_params:,}")
        print(f"Network trainable parameters: {trainable_params:,}")