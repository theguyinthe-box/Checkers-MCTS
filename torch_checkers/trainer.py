"""
###############################################################################
# torch/trainer.py
#
# Training loop for Checkers MCTS neural network.
#
# This module implements the training pipeline including:
# - Mixed precision training for GPU efficiency
# - Gradient accumulation for effective larger batch sizes
# - Learning rate scheduling (cosine, cyclic, step)
# - Early stopping based on validation loss
# - Checkpoint saving and loading
# - Training metrics logging
#
# GPU Optimization:
# - Uses torch.cuda.amp for automatic mixed precision
# - Gradient checkpointing option for memory efficiency
# - Efficient data loading with pin_memory and prefetching
#
###############################################################################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, Optional, Tuple, List, Any
import os
import time
from datetime import datetime
import numpy as np

from .config import Config
from .model import CheckersModel
from .dataset import CheckersDataset, create_data_loaders, collate_fn
from .utils import (
    save_checkpoint, 
    load_checkpoint, 
    setup_logging,
    AverageMeter,
    EarlyStopping,
    parse_device_type
)


class Trainer:
    """
    Trainer class for the Checkers neural network.
    
    This class handles the complete training pipeline including:
    - Model compilation and optimization setup
    - Training and validation loops
    - Learning rate scheduling
    - Mixed precision training
    - Checkpointing and logging
    
    The trainer is optimized for 16GB VRAM with options for
    gradient accumulation to simulate larger batch sizes.
    
    Args:
        model: CheckersModel instance.
        config: Configuration object.
        
    Example:
        >>> config = Config()
        >>> model = CheckersModel(config)
        >>> trainer = Trainer(model, config)
        >>> trainer.train(train_data, num_epochs=100)
    """
    
    def __init__(
        self,
        model: CheckersModel,
        config: Config
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model.
            config: Configuration object.
        """
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision training
        device_type = parse_device_type(config.device)
        self.scaler = GradScaler(device_type) if config.mixed_precision and device_type == 'cuda' else None
        self.use_amp = config.mixed_precision and device_type == 'cuda'
        self.device_type = device_type  # Store for later use in autocast
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Loss weights
        self.policy_weight = config.policy_loss_weight
        self.value_weight = config.value_loss_weight
        
        # Gradient accumulation
        self.accumulation_steps = config.gradient_accumulation_steps
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(config.log_dir)
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler based on config.
        
        Returns:
            PyTorch learning rate scheduler.
        """
        config = self.config
        
        if config.lr_schedule == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.min_learning_rate
            )
        elif config.lr_schedule == 'cyclic':
            return torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=config.cyclic_base_lr,
                max_lr=config.cyclic_max_lr,
                step_size_up=2000,
                mode='triangular'
            )
        elif config.lr_schedule == 'step':
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=config.step_lr_milestones,
                gamma=config.step_lr_gamma
            )
        else:
            # Default: constant LR
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
    
    def train(
        self,
        train_data: List[Tuple[np.ndarray, np.ndarray, float, float]],
        num_epochs: Optional[int] = None,
        val_split: Optional[float] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model on provided data.
        
        Args:
            train_data: List of training examples.
            num_epochs: Number of epochs (uses config if None).
            val_split: Validation split fraction (uses config if None).
            
        Returns:
            Dictionary containing training history.
        """
        num_epochs = num_epochs or self.config.num_epochs
        val_split = val_split if val_split is not None else self.config.validation_split
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_data,
            val_split=val_split,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            augment=self.config.augment_data
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_policy_loss': [],
            'train_value_loss': [],
            'val_loss': [],
            'val_policy_loss': [],
            'val_value_loss': [],
            'learning_rate': []
        }
        
        # Calculate epoch range for this training session
        start_epoch = self.current_epoch
        end_epoch = start_epoch + num_epochs
        
        self.logger.info(f"Starting training for {num_epochs} epochs (global epochs {start_epoch + 1} to {end_epoch})")
        self.logger.info(f"Training samples: {len(train_data)}, Batch size: {self.config.batch_size}")
        self.logger.info(f"Device: {self.device}, Mixed precision: {self.use_amp}")
        
        # Import tqdm for progress bar
        from tqdm import tqdm
        
        # Determine if progress should be shown
        show_progress = getattr(self.config, 'show_progress', True)
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(start_epoch, end_epoch),
            desc="Training epochs",
            disable=not show_progress,
            unit="epoch"
        )
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, show_progress=show_progress)
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
            else:
                val_metrics = {'loss': 0, 'policy_loss': 0, 'value_loss': 0}
            
            # Update learning rate
            if self.config.lr_schedule != 'cyclic':
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_metrics['loss'])
            history['train_policy_loss'].append(train_metrics['policy_loss'])
            history['train_value_loss'].append(train_metrics['value_loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_policy_loss'].append(val_metrics['policy_loss'])
            history['val_value_loss'].append(val_metrics['value_loss'])
            history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train': f"{train_metrics['loss']:.4f}",
                'val': f"{val_metrics['loss']:.4f}",
                'LR': f"{current_lr:.2e}"
            })
            
            # Log progress (only when progress bars are disabled)
            if not show_progress:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"(P: {train_metrics['policy_loss']:.4f}, V: {train_metrics['value_loss']:.4f}) - "
                    f"Val Loss: {val_metrics['loss']:.4f} - "
                    f"LR: {current_lr:.2e} - "
                    f"Time: {epoch_time:.1f}s"
                )
            
            # Save checkpoint if best
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_best_model(epoch, val_metrics['loss'])
            
            # Regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping check
            if val_loader is not None and self.early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                self.current_epoch = epoch + 1  # Update here for early stopping case
                break
        else:
            # Loop completed without early stopping - update to next epoch
            self.current_epoch = end_epoch
        
        return history
    
    def _train_epoch(self, data_loader: DataLoader, show_progress: bool = True) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Args:
            data_loader: Training data loader.
            show_progress: Whether to show batch progress bar.
            
        Returns:
            Dictionary with average loss values.
        """
        from tqdm import tqdm
        
        self.model.train()
        
        loss_meter = AverageMeter()
        policy_loss_meter = AverageMeter()
        value_loss_meter = AverageMeter()
        
        self.optimizer.zero_grad()
        
        # Create progress bar for batches
        batch_pbar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="  Batches",
            disable=not show_progress,
            leave=False,
            unit="batch"
        )
        
        for batch_idx, batch in batch_pbar:
            # Move data to device
            states = batch['state'].to(self.device)
            target_policies = batch['policy'].to(self.device)
            target_values = batch['value'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(device_type=self.device_type, enabled=self.use_amp):
                policy_logits, value_preds = self.model(states)
                
                # Policy loss (cross-entropy)
                policy_loss = F.cross_entropy(
                    policy_logits, 
                    target_policies,
                    reduction='mean'
                )
                
                # Value loss (MSE)
                value_loss = F.mse_loss(
                    value_preds, 
                    target_values,
                    reduction='mean'
                )
                
                # Combined loss
                loss = (
                    self.policy_weight * policy_loss + 
                    self.value_weight * value_loss
                )
                
                # Scale for gradient accumulation
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step (with accumulation)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Cyclic LR updates per step
                if self.config.lr_schedule == 'cyclic':
                    self.scheduler.step()
            
            # Update meters
            batch_size = states.size(0)
            loss_meter.update(loss.item() * self.accumulation_steps, batch_size)
            policy_loss_meter.update(policy_loss.item(), batch_size)
            value_loss_meter.update(value_loss.item(), batch_size)
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'p_loss': f"{policy_loss_meter.avg:.4f}",
                'v_loss': f"{value_loss_meter.avg:.4f}"
            })
            
            # Log interval (for when progress is disabled)
            if not show_progress and batch_idx % self.config.log_interval == 0:
                self.logger.debug(
                    f"Batch {batch_idx}/{len(data_loader)} - "
                    f"Loss: {loss_meter.avg:.4f}"
                )
        
        return {
            'loss': loss_meter.avg,
            'policy_loss': policy_loss_meter.avg,
            'value_loss': value_loss_meter.avg
        }
    
    def _validate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation.
        
        Args:
            data_loader: Validation data loader.
            
        Returns:
            Dictionary with average loss values.
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        policy_loss_meter = AverageMeter()
        value_loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in data_loader:
                states = batch['state'].to(self.device)
                target_policies = batch['policy'].to(self.device)
                target_values = batch['value'].to(self.device)
                
                with autocast(device_type=self.device_type, enabled=self.use_amp):
                    policy_logits, value_preds = self.model(states)
                    
                    policy_loss = F.cross_entropy(
                        policy_logits, target_policies
                    )
                    value_loss = F.mse_loss(value_preds, target_values)
                    loss = (
                        self.policy_weight * policy_loss + 
                        self.value_weight * value_loss
                    )
                
                batch_size = states.size(0)
                loss_meter.update(loss.item(), batch_size)
                policy_loss_meter.update(policy_loss.item(), batch_size)
                value_loss_meter.update(value_loss.item(), batch_size)
        
        return {
            'loss': loss_meter.avg,
            'policy_loss': policy_loss_meter.avg,
            'value_loss': value_loss_meter.avg
        }
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch+1}.pt'
        )
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint: {filepath}")
    
    def _save_best_model(self, epoch: int, val_loss: float):
        """Save best model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict()
        }
        
        filepath = os.path.join(
            self.config.checkpoint_dir,
            'best_model.pt'
        )
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")


def train_from_self_play(
    model: CheckersModel,
    replay_buffer,
    config: Config,
    num_epochs: int = 1
) -> Dict[str, float]:
    """
    Train the model from self-play data in replay buffer.
    
    This function is used during the self-play training loop to
    train the model on recent game experiences.
    
    Args:
        model: Neural network model.
        replay_buffer: Replay buffer with training data.
        config: Configuration object.
        num_epochs: Number of training epochs.
        
    Returns:
        Dictionary with training metrics.
    """
    trainer = Trainer(model, config)
    
    # Get data from replay buffer
    train_data = replay_buffer.get_all()
    
    if len(train_data) < config.batch_size:
        return {'loss': 0, 'policy_loss': 0, 'value_loss': 0}
    
    # Train for specified epochs
    history = trainer.train(train_data, num_epochs=num_epochs)
    
    # Return final metrics
    return {
        'loss': history['train_loss'][-1] if history['train_loss'] else 0,
        'policy_loss': history['train_policy_loss'][-1] if history['train_policy_loss'] else 0,
        'value_loss': history['train_value_loss'][-1] if history['train_value_loss'] else 0
    }
