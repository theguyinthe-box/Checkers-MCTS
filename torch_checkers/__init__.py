"""
###############################################################################
# torch_checkers/__init__.py
#
# PyTorch implementation of MCTS for Checkers with neural network guidance.
#
# This module provides GPU-optimized training and evaluation of MCTS with
# a neural network for playing Checkers. The implementation follows the
# AlphaZero approach with policy and value heads.
#
# Modules:
# - model: Neural network architecture (ResNet-based policy-value network)
# - mcts: Monte Carlo Tree Search with batched GPU inference
# - dataset: Custom PyTorch Dataset for efficient data loading
# - trainer: Training loop with mixed precision and gradient accumulation
# - config: Configuration dataclass for hyperparameters
# - utils: Utility functions for logging, checkpointing, etc.
#
# Usage:
#   from torch_checkers import CheckersModel, MCTSPlayer, Trainer, Config
#
###############################################################################
"""

from .config import Config
from .model import CheckersModel
from .mcts import MCTSPlayer, MCTSNode
from .dataset import CheckersDataset
from .trainer import Trainer
from .utils import setup_logging, save_checkpoint, load_checkpoint

__all__ = [
    'Config',
    'CheckersModel',
    'MCTSPlayer',
    'MCTSNode',
    'CheckersDataset',
    'Trainer',
    'setup_logging',
    'save_checkpoint',
    'load_checkpoint',
]

__version__ = '1.0.0'
