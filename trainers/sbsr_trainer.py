import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from tqdm import tqdm
import lpips
from .base_trainer import BaseTrainer
from utils import Visualizer

class SBSRTrainer(BaseTrainer):
    """Trainer for SBSR model."""
    
    def __init__(self, model: nn.Module, config: Any):
        super().__init__(model, config)
        
        # Initialize loss functions
        self.setup_loss_functions()
        
        # Create results directory
        os.makedirs(self.config.sample_dir, exist_ok=True)
        
    def setup_loss_functions(self):
        """Setup all loss functions."""
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
        
        # VGG loss
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_layers = nn.Sequential(*list(vgg)[:36]).to(self.device)
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
            
    def compute_vgg_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute VGG perceptual loss."""
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return F.l1_loss(x_vgg, y_vgg)
        
    def compute_focal_frequency_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Focal Frequency Loss."""
        x_fft = torch.fft.fft2(x)
        y_fft = torch.fft.fft2(y)
        diff = x_fft - y_fft
        return torch.mean(torch.abs(diff) ** self.config.ff_alpha)
        
    def compute_total_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}
        
        # Basic losses
        losses['l1'] = self.l1_loss(pred, target)
        losses['vgg'] = self.compute_vgg_loss(pred, target) * self.config.lambda_vgg
        losses['ff'] = self.compute_focal_frequency_loss(pred, target) * self.config.lambda_ff
        losses['lpips'] = self.lpips_loss(pred, target).mean()
        
        # Total loss
        losses['total'] = losses['l1'] + losses['vgg'] + losses['ff']
        
        return losses
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()
        
        epoch_losses = {}
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {self.current_epoch + 1}/{self.config.num_epochs}')
        
        for batch_idx, (lq_images, gt_images) in enumerate(train_loader):
            lq_images = lq_images.to(self.device)
            gt_images = gt_images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_images = self.model(lq_images)
            
            # Compute losses
            losses = self.compute_total_loss(pred_images, gt_images)
            
            # Backward pass
            losses['total'].backward()
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
            self.optimizer.step()
            
            # Update metrics
            self.metric_tracker.update(pred_images, gt_images, losses['l1'].item())
            
            # Update progress bar
            pbar.set_postfix(**{k: v.item() for k, v in losses.items()})
            pbar.update(1)
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v.item()
                
            # Log batch results
            if batch_idx % self.config.log_interval == 0:
                self.logger.log_images({
                    'input': lq_images,
                    'pred': pred_images,
                    'gt': gt_images
                }, self.current_epoch * len(train_loader) + batch_idx)
                
        pbar.close()
        
        # Average losses
        epoch_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        
        return epoch_losses
        
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        self.metric_tracker.reset()
        
        val_losses = {}
        
        with torch.no_grad():
            for batch_idx, (lq_images, gt_images) in enumerate(val_loader):
                lq_images = lq_images.to(self.device)
                gt_images = gt_images.to(self.device)
                
                # Forward pass
                pred_images = self.model(lq_images)
                
                # Compute losses
                losses = self.compute_total_loss(pred_images, gt_images)
                
                # Update metrics
                self.metric_tracker.update(pred_images, gt_images, losses['l1'].item())
                
                # Accumulate losses
                for k, v in losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v.item()
                    
                # Save validation samples
                if batch_idx == 0:
                    Visualizer.visualize_batch(
                        lq_images, pred_images, gt_images,
                        save_path=os.path.join(
                            self.config.sample_dir,
                            f'val_epoch_{self.current_epoch}.png'
                        )
                    )
                    
        # Average losses
        val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
        
        return val_losses
        
    def train(self, train_loader, val_loader):
        """Full training loop."""
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_losses = self.train_epoch(train_loader)
            self.logger.log_metrics(train_losses, epoch, 'train')
            
            # Validation phase
            if epoch % self.config.val_interval == 0:
                val_losses = self.validate(val_loader)
                self.logger.log_metrics(val_losses, epoch, 'val')
                
                # Update learning rate
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
                
                # Save checkpoint if best so far
                if val_losses['total'] < self.best_metric:
                    self.best_metric = val_losses['total']
                    self.save_checkpoint(epoch, val_losses['total'], is_best=True)
                    
            # Regular checkpoint saving
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, train_losses['total'])
                
        self.logger.close()