import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional

class TensorboardLogger:
    """Tensorboard logging utility"""
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize Tensorboard logger.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of the experiment (default: timestamp)
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_dir)
        
    def log_scalars(self, tag: str, scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars with the same tag."""
        self.writer.add_scalars(tag, scalar_dict, step)
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        self.writer.add_scalar(tag, value, step)
        
    def log_images(self, tag: str, images: torch.Tensor, step: int):
        """Log images to tensorboard."""
        self.writer.add_images(tag, images, step)
        
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        self.writer.add_histogram(tag, values, step)
        
    def close(self):
        """Close the tensorboard writer."""
        self.writer.close()

class Logger:
    """General purpose logger combining file and tensorboard logging"""
    
    def __init__(self, 
                 config,
                 log_dir: str = "logs",
                 experiment_name: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            config: Configuration object
            log_dir: Directory for logs
            experiment_name: Name of the experiment
        """
        self.config = config
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        self.log_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize tensorboard
        self.tb = TensorboardLogger(log_dir, self.experiment_name)
        
        # Setup file logging
        self.setup_file_logger()
        
    def setup_file_logger(self):
        """Setup file logging configuration."""
        log_file = os.path.join(self.log_dir, 'training.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = 'train'):
        """
        Log metrics to both tensorboard and file.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch
            phase: 'train' or 'val'
        """
        # Log to tensorboard
        self.tb.log_scalars(f'{phase}/metrics', metrics, step)
        
        # Log to file
        metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        logging.info(f'{phase.capitalize()} Epoch {step} - {metrics_str}')
        
    def log_images(self, 
                  images_dict: Dict[str, torch.Tensor],
                  step: int,
                  phase: str = 'train'):
        """
        Log images to tensorboard.
        
        Args:
            images_dict: Dictionary of image names and tensors
            step: Current step/epoch
            phase: 'train' or 'val'
        """
        for name, images in images_dict.items():
            self.tb.log_images(f'{phase}/{name}', images, step)
            
    def log_model_grad_stats(self, model: torch.nn.Module, step: int):
        """Log model gradient statistics."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.tb.log_histogram(f'gradients/{name}', param.grad, step)
                
    def log_config(self):
        """Log configuration parameters."""
        logging.info("Configuration:")
        for key, value in vars(self.config).items():
            logging.info(f"{key}: {value}")
            
    def close(self):
        """Close all logging handlers."""
        self.tb.close()