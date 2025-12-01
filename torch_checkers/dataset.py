"""
###############################################################################
# torch/dataset.py
#
# PyTorch Dataset for Checkers MCTS training data.
#
# This module provides efficient data loading for training the neural network.
# Features include:
# - Memory-efficient data handling with memory mapping
# - Board symmetry augmentation for data efficiency
# - Custom collate function for batching
# - Support for streaming data from self-play
#
###############################################################################
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Tuple, Optional, Dict, Any
import pickle
import os
from collections import deque
import random


class CheckersDataset(Dataset):
    """
    PyTorch Dataset for Checkers training data.
    
    This dataset stores training examples from self-play games and
    provides efficient access for training. Each example contains:
    - State: Board representation (14 channels, 8x8)
    - Policy: Target policy from MCTS (8 directions, 8x8)
    - Value: Target value (Q-value and game outcome average)
    
    The dataset supports optional data augmentation through board
    symmetry transformations.
    
    Args:
        data: List of training examples.
        augment: Whether to use data augmentation.
        transform: Optional additional transforms.
    
    Example:
        >>> data = [(state, policy, q_val, z_val), ...]
        >>> dataset = CheckersDataset(data, augment=True)
        >>> loader = DataLoader(dataset, batch_size=64, shuffle=True)
    """
    
    def __init__(
        self,
        data: Optional[List[Tuple[np.ndarray, np.ndarray, float, float]]] = None,
        augment: bool = False,
        transform: Optional[Any] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data: List of (state, policy, q_value, z_value) tuples.
            augment: Whether to apply data augmentation.
            transform: Optional transforms to apply.
        """
        self.data = data if data is not None else []
        self.augment = augment
        self.transform = transform
        
        # Pre-compute augmented indices if augmenting
        if self.augment:
            self._augmented_length = len(self.data) * 2  # Horizontal flip only for Checkers
        else:
            self._augmented_length = len(self.data)
    
    def __len__(self) -> int:
        """Return number of examples (including augmentations)."""
        return self._augmented_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example.
        
        Args:
            idx: Index of example to retrieve.
            
        Returns:
            Dictionary with:
                - 'state': Tensor of shape (14, 8, 8)
                - 'policy': Tensor of shape (512,)
                - 'value': Tensor of shape (1,)
        """
        if self.augment:
            # Determine if this is an augmented version
            base_idx = idx % len(self.data)
            aug_type = idx // len(self.data)
        else:
            base_idx = idx
            aug_type = 0
        
        state, policy, q_value, z_value = self.data[base_idx]
        
        # Apply augmentation if needed
        if aug_type == 1:
            state, policy = self._horizontal_flip(state, policy)
        
        # Convert to tensors
        state_tensor = torch.from_numpy(state.copy()).float()
        policy_tensor = torch.from_numpy(policy.flatten().copy()).float()
        
        # Value is average of Q-value and game outcome
        value = (q_value + z_value) / 2.0
        value_tensor = torch.tensor([value], dtype=torch.float32)
        
        # Apply additional transforms if any
        if self.transform is not None:
            state_tensor = self.transform(state_tensor)
        
        return {
            'state': state_tensor,
            'policy': policy_tensor,
            'value': value_tensor
        }
    
    def _horizontal_flip(
        self,
        state: np.ndarray,
        policy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply horizontal flip augmentation.
        
        For Checkers, we can flip the board horizontally. This requires
        swapping left/right moves in the policy.
        
        Args:
            state: State array of shape (14, 8, 8).
            policy: Policy array of shape (8, 8, 8).
            
        Returns:
            Flipped state and policy arrays.
        """
        # Flip state horizontally
        flipped_state = np.flip(state, axis=2).copy()
        
        # Flip policy and swap left/right directions
        # Policy layout: [UL, UR, BL, BR, UL_jump, UR_jump, BL_jump, BR_jump]
        # After flip: UL<->UR, BL<->BR
        flipped_policy = np.flip(policy, axis=2).copy()
        
        # Swap direction pairs
        # UL (0) <-> UR (1)
        # BL (2) <-> BR (3)
        # UL_jump (4) <-> UR_jump (5)
        # BL_jump (6) <-> BR_jump (7)
        swap_indices = [1, 0, 3, 2, 5, 4, 7, 6]
        flipped_policy = flipped_policy[swap_indices]
        
        return flipped_state, flipped_policy
    
    def add_data(self, new_data: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        """
        Add new training data to the dataset.
        
        Args:
            new_data: List of new training examples.
        """
        self.data.extend(new_data)
        if self.augment:
            self._augmented_length = len(self.data) * 2
        else:
            self._augmented_length = len(self.data)
    
    def clear(self):
        """Clear all data from the dataset."""
        self.data = []
        self._augmented_length = 0
    
    def save(self, filepath: str):
        """
        Save dataset to file.
        
        Args:
            filepath: Path to save file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f)
    
    @classmethod
    def load(cls, filepath: str, augment: bool = False) -> 'CheckersDataset':
        """
        Load dataset from file.
        
        Args:
            filepath: Path to saved file.
            augment: Whether to enable augmentation.
            
        Returns:
            Loaded CheckersDataset instance.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(data=data, augment=augment)


class ReplayBuffer:
    """
    Experience replay buffer for self-play training.
    
    This buffer maintains a fixed-size collection of training examples
    from self-play games. It supports efficient sampling for training
    and automatic eviction of old examples.
    
    Args:
        max_size: Maximum number of examples to store.
        
    Example:
        >>> buffer = ReplayBuffer(max_size=100000)
        >>> buffer.add(game_data)
        >>> batch = buffer.sample(batch_size=64)
    """
    
    def __init__(self, max_size: int = 100000):
        """Initialize replay buffer."""
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def add(self, experiences: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        """
        Add experiences to the buffer.
        
        Args:
            experiences: List of (state, policy, q_value, z_value) tuples.
        """
        for exp in experiences:
            self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of examples to sample.
            
        Returns:
            List of sampled experiences.
        """
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)
    
    def get_all(self) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        """Return all experiences in the buffer."""
        return list(self.buffer)
    
    def clear(self):
        """Clear all experiences from the buffer."""
        self.buffer.clear()
    
    def save(self, filepath: str):
        """Save buffer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, filepath: str):
        """Load buffer from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.buffer = deque(data, maxlen=self.max_size)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Combines individual samples into batched tensors efficiently.
    
    Args:
        batch: List of sample dictionaries from dataset.
        
    Returns:
        Batched dictionary with stacked tensors.
    """
    states = torch.stack([item['state'] for item in batch])
    policies = torch.stack([item['policy'] for item in batch])
    values = torch.stack([item['value'] for item in batch])
    
    return {
        'state': states,
        'policy': policies,
        'value': values
    }


def create_data_loaders(
    train_data: List[Tuple[np.ndarray, np.ndarray, float, float]],
    val_split: float = 0.2,
    batch_size: int = 256,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation DataLoaders.
    
    Args:
        train_data: List of training examples.
        val_split: Fraction of data for validation.
        batch_size: Batch size for loaders.
        num_workers: Number of data loading workers.
        augment: Whether to use data augmentation.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Shuffle data
    random.shuffle(train_data)
    
    # Split data
    if val_split > 0:
        split_idx = int(len(train_data) * (1 - val_split))
        train_split = train_data[:split_idx]
        val_split_data = train_data[split_idx:]
    else:
        train_split = train_data
        val_split_data = None
    
    # Create datasets
    train_dataset = CheckersDataset(train_split, augment=augment)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    if val_split_data is not None and len(val_split_data) > 0:
        val_dataset = CheckersDataset(val_split_data, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        val_loader = None
    
    return train_loader, val_loader


class StreamingDataset(IterableDataset):
    """
    Iterable dataset for streaming self-play data.
    
    This dataset is used when self-play and training happen
    concurrently. It yields examples from a shared replay buffer.
    
    Args:
        replay_buffer: Shared replay buffer instance.
        batch_size: Number of examples per iteration.
    """
    
    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        """Initialize streaming dataset."""
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
    
    def __iter__(self):
        """Iterate over batches from replay buffer."""
        while True:
            if len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                for state, policy, q_val, z_val in batch:
                    yield {
                        'state': torch.from_numpy(state.copy()).float(),
                        'policy': torch.from_numpy(policy.flatten().copy()).float(),
                        'value': torch.tensor([(q_val + z_val) / 2.0], dtype=torch.float32)
                    }
            else:
                # Not enough data yet
                break
