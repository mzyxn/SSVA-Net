import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

class MetricTracker:
    """Track and compute various image quality metrics"""
    
    def __init__(self, device: torch.device):
        """
        Initialize metric tracker.
        
        Args:
            device: Torch device for computations
        """
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.reset()
        
    def reset(self):
        """Reset all metric counters."""
        self.metrics = {
            'l1_loss': [],
            'psnr': [],
            'ssim': [],
            'lpips': []
        }
        
    def update(self, pred: torch.Tensor, target: torch.Tensor, l1_loss: float):
        """
        Update metrics with new batch.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            l1_loss: L1 loss value
        """
        # Convert tensors to numpy for CPU metrics
        pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
        target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
        
        # Update L1 loss
        self.metrics['l1_loss'].append(l1_loss)
        
        # Compute PSNR and SSIM for each image in batch
        for p, t in zip(pred_np, target_np):
            self.metrics['psnr'].append(psnr(t, p, data_range=1.0))
            self.metrics['ssim'].append(ssim(t, p, data_range=1.0, channel_axis=-1))
        
        # Compute LPIPS
        with torch.no_grad():
            lpips_value = self.lpips_fn(pred, target).mean().item()
            self.metrics['lpips'].append(lpips_value)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get average of all metrics."""
        return {
            'l1_loss': np.mean(self.metrics['l1_loss']),
            'psnr': np.mean(self.metrics['psnr']),
            'ssim': np.mean(self.metrics['ssim']),
            'lpips': np.mean(self.metrics['lpips'])
        }

def compute_masked_metrics(pred: torch.Tensor, 
                         target: torch.Tensor, 
                         mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics only in masked regions.
    
    Args:
        pred: Predicted images (B, C, H, W)
        target: Target images (B, C, H, W)
        mask: Binary mask (B, 1, H, W)
        
    Returns:
        Dictionary of metrics
    """
    # Apply mask
    pred_masked = pred * mask
    target_masked = target * mask
    
    # Convert to numpy
    pred_np = pred_masked.detach().cpu().numpy().transpose(0, 2, 3, 1)
    target_np = target_masked.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Compute metrics
    metrics = {}
    for i in range(pred_np.shape[0]):
        metrics[f'psnr_{i}'] = psnr(target_np[i], pred_np[i], data_range=1.0)
        metrics[f'ssim_{i}'] = ssim(target_np[i], pred_np[i], data_range=1.0, 
                                  channel_axis=-1)
    
    # Average metrics
    metrics['psnr_avg'] = np.mean([v for k, v in metrics.items() if 'psnr' in k])
    metrics['ssim_avg'] = np.mean([v for k, v in metrics.items() if 'ssim' in k])
    
    return metrics