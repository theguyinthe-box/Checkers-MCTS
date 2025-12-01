"""
###############################################################################
# torch/utils.py
#
# Utility functions for Checkers MCTS training.
#
# This module provides helper functions for:
# - Logging setup and management
# - Checkpoint saving/loading
# - Training metrics tracking
# - Visualization and plotting
# - Early stopping implementation
# - Random seed management
#
###############################################################################
"""

import logging
import os
import sys
import torch
import numpy as np
import random
from datetime import datetime
from typing import Optional, Dict, Any, List
import json


def setup_logging(
    log_dir: str,
    log_level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Creates a logger that writes to both console and file.
    
    Args:
        log_dir: Directory for log files.
        log_level: Logging level (DEBUG, INFO, etc.).
        log_to_file: Whether to write logs to file.
        
    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('checkers_mcts')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def parse_device_type(device: str) -> str:
    """
    Parse device string to extract device type.
    
    Handles device strings like 'cuda', 'cuda:0', 'cpu', etc.
    
    Args:
        device: Device string (e.g., 'cuda:0', 'cpu').
        
    Returns:
        Device type string (e.g., 'cuda', 'cpu').
    """
    return device.split(':')[0] if ':' in device else device

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Sets seed for Python, NumPy, and PyTorch random number generators.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_val_loss: float = float('inf'),
    config: Optional[Dict] = None,
    **kwargs
):
    """
    Save training checkpoint.
    
    Args:
        filepath: Path to save checkpoint.
        model: PyTorch model.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        epoch: Current epoch number.
        best_val_loss: Best validation loss so far.
        config: Configuration dictionary.
        **kwargs: Additional items to save.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file.
        model: PyTorch model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        device: Device to load tensors to.
        
    Returns:
        Dictionary with checkpoint metadata.
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


class AverageMeter:
    """
    Computes and stores running average and current value.
    
    Used for tracking training metrics like loss.
    
    Example:
        >>> meter = AverageMeter()
        >>> for i in range(10):
        ...     meter.update(loss_value, batch_size)
        >>> print(meter.avg)
    """
    
    def __init__(self):
        """Initialize meter."""
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: New value to add.
            n: Number of samples this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for loss, 'max' for accuracy.
        
    Example:
        >>> early_stop = EarlyStopping(patience=10, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stop(val_loss):
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value (usually validation loss).
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.early_stop = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory usage information.
    
    Returns:
        Dictionary with memory info in GB.
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'total': 0}
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'total': torch.cuda.get_device_properties(0).total_memory / 1e9
    }


def create_timestamp() -> str:
    """
    Create timestamp string for filenames.
    
    Returns:
        Timestamp string in format 'YYYYMMDD_HHMMSS'.
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_training_history(
    history: Dict[str, List[float]],
    filepath: str
):
    """
    Save training history to JSON file.
    
    Args:
        history: Dictionary of training metrics.
        filepath: Path to save file.
    """
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)


def load_training_history(filepath: str) -> Dict[str, List[float]]:
    """
    Load training history from JSON file.
    
    Args:
        filepath: Path to history file.
        
    Returns:
        Dictionary of training metrics.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary of training metrics.
        save_path: Optional path to save plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot losses
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    if 'train_loss' in history and history['train_loss']:
        axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    if 'train_policy_loss' in history and history['train_policy_loss']:
        axes[1].plot(epochs, history['train_policy_loss'], label='Train')
    if 'val_policy_loss' in history and history['val_policy_loss']:
        axes[1].plot(epochs, history['val_policy_loss'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Policy Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    if 'train_value_loss' in history and history['train_value_loss']:
        axes[2].plot(epochs, history['train_value_loss'], label='Train')
    if 'val_value_loss' in history and history['val_value_loss']:
        axes[2].plot(epochs, history['val_value_loss'], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Value Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.close()


class ProgressBar:
    """
    Simple text-based progress bar.
    
    Args:
        total: Total number of iterations.
        prefix: Prefix string.
        length: Length of progress bar.
    """
    
    def __init__(
        self,
        total: int,
        prefix: str = '',
        length: int = 40
    ):
        """Initialize progress bar."""
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
    
    def update(self, n: int = 1, info: str = ''):
        """
        Update progress bar.
        
        Args:
            n: Number of steps to advance.
            info: Additional info to display.
        """
        self.current += n
        filled = int(self.length * self.current / self.total)
        bar = '█' * filled + '-' * (self.length - filled)
        percent = 100 * self.current / self.total
        print(f'\r{self.prefix} |{bar}| {percent:.1f}% {info}', end='')
        if self.current >= self.total:
            print()
    
    def reset(self):
        """Reset progress bar."""
        self.current = 0
