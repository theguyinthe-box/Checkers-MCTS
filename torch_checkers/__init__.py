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
# - random_player: Random Legal Move player for benchmarking
# - experiment_tournament: Multi-player tournament training experiment
# - experiment_single: Single player training with evaluation
# - evaluate_vs_random: Evaluation pipeline for testing against random player
#
# Usage:
#   from torch_checkers import CheckersModel, MCTSPlayer, Trainer, Config
#   from torch_checkers import RandomPlayer
#
###############################################################################
"""

from .config import Config
from .model import CheckersModel
from .mcts import MCTSPlayer, MCTSNode
from .dataset import CheckersDataset
from .trainer import Trainer
from .utils import setup_logging, save_checkpoint, load_checkpoint, parse_device_type
from .random_player import RandomPlayer

__all__ = [
    'Config',
    'CheckersModel',
    'MCTSPlayer',
    'MCTSNode',
    'CheckersDataset',
    'Trainer',
    'RandomPlayer',
    'setup_logging',
    'save_checkpoint',
    'load_checkpoint',
    'parse_device_type',
]

__version__ = '1.0.0'
