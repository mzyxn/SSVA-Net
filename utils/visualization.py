import torch
import torchvision.utils as vutils
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """Utility class for visualization of images and results"""
    
    @staticmethod
    def create_grid(images: List[torch.Tensor], 
                   nrow: int = 8,
                   padding: int = 2) -> torch.Tensor:
        """Create a grid of images."""
        return vutils.make_grid(images, nrow=nrow, padding=padding, normalize=True)
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for visualization."""
        if tensor.ndim == 4:
            return tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)
    
    @staticmethod
    def visualize_batch(lq_images: torch.Tensor,
                       pred_images: torch.Tensor,
                       gt_images: torch.Tensor,
                       save_path: Optional[str] = None,
                       max_samples: int = 8):
        """
        Visualize a batch of images (LQ, Pred, GT).
        
        Args:
            lq_images: Low quality images
            pred_images: Predicted images
            gt_images: Ground truth images
            save_path: Path to save visualization
            max_samples: Maximum number of samples to visualize
        """
        # Take first max_samples images
        lq_images = lq_images[:max_samples]
        pred_images = pred_images[:max_samples]
        gt_images = gt_images[:max_samples]
        
        # Create figure
        fig, axes = plt.subplots(max_samples, 3, figsize=(15, 5*max_samples))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        for i in range(max_samples):
            # Convert tensors to numpy
            lq = Visualizer.tensor_to_numpy(lq_images[i])
            pred = Visualizer.tensor_to_numpy(pred_images[i])
            gt = Visualizer.tensor_to_numpy(gt_images[i])
            
            # Plot images
            axes[i, 0].imshow(lq)
            axes[i, 0].set_title('Input')
            axes[i, 1].imshow(pred)
            axes[i, 1].set_title('Prediction')
            axes[i, 2].imshow(gt)
            axes[i, 2].set_title('Ground Truth')
            
            # Remove axes
            for ax in axes[i]:
                ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_attention_maps(attention_maps: torch.Tensor,
                          save_path: Optional[str] = None):
        """
        Visualize attention maps.
        
        Args:
            attention_maps: Attention maps tensor (B, H, W)
            save_path: Path to save visualization
        """
        attention_maps = attention_maps.detach().cpu()
        
        fig, axes = plt.subplots(1, attention_maps.size(0), 
                                figsize=(4*attention_maps.size(0), 4))
        if attention_maps.size(0) == 1:
            axes = [axes]
            
        for i, attention in enumerate(attention_maps):
            im = axes[i].imshow(attention, cmap='viridis')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
            
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()