"""
###############################################################################
# torch/config.py
#
# Configuration dataclass for PyTorch MCTS Checkers implementation.
#
# This module provides a centralized configuration system using Python
# dataclasses. All hyperparameters for training, MCTS, and neural network
# architecture are defined here for easy modification and reproducibility.
#
# GPU Memory Budget: Optimized for 16GB VRAM
# - Batch sizes and model parameters are tuned accordingly
# - Mixed precision training is enabled by default
#
###############################################################################
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class Config:
    """
    Configuration class for Checkers MCTS with neural network.
    
    This dataclass contains all hyperparameters needed for training,
    evaluation, and playing. Parameters are grouped by functionality
    and optimized for a 16GB VRAM budget.
    
    Attributes:
        # Model Architecture
        num_res_blocks: Number of residual blocks in the network backbone.
        num_channels: Number of channels in convolutional layers.
        input_channels: Number of input feature planes (from Checkers state).
        board_size: Size of the board (8 for standard Checkers).
        policy_output_size: Number of possible moves (8 directions * 8 * 8).
        
        # Training Parameters
        batch_size: Training batch size (tuned for 16GB VRAM).
        learning_rate: Base learning rate for optimizer.
        weight_decay: L2 regularization coefficient.
        num_epochs: Maximum number of training epochs.
        warmup_epochs: Number of epochs for learning rate warmup.
        
        # MCTS Parameters
        num_simulations: Number of MCTS simulations per move.
        cpuct: Exploration constant in PUCT formula.
        temperature: Temperature for move selection during self-play.
        temperature_drop_move: Move number to drop temperature to near-zero.
        dirichlet_alpha: Dirichlet noise parameter for exploration.
        dirichlet_epsilon: Fraction of Dirichlet noise added to root.
        
        # Self-Play Parameters
        num_self_play_games: Games per self-play iteration.
        max_game_moves: Maximum moves before terminating a game.
        
        # Evaluation Parameters
        num_evaluation_games: Games to play during evaluation.
        win_threshold: Win rate threshold to accept new model.
        
        # System Parameters
        device: PyTorch device ('cuda' or 'cpu').
        num_workers: Number of data loading workers.
        mixed_precision: Enable automatic mixed precision training.
        gradient_accumulation_steps: Steps for gradient accumulation.
        checkpoint_dir: Directory for saving checkpoints.
        data_dir: Directory for training data.
        log_dir: Directory for training logs.
    """
    
    # Model Architecture
    num_res_blocks: int = 10
    num_channels: int = 128
    input_channels: int = 14  # Channels 0-13 from Checkers state (excluding channel 14)
    board_size: int = 8
    num_action_directions: int = 8  # 4 normal moves + 4 jumps
    policy_output_size: int = 512  # 8 directions * 8 * 8 positions = 512
    
    # Training Parameters
    batch_size: int = 256  # Optimized for 16GB VRAM with mixed precision
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 5
    patience: int = 20  # Early stopping patience
    min_delta: float = 0.001  # Minimum improvement for early stopping
    
    # Learning Rate Schedule
    lr_schedule: str = 'cosine'  # Options: 'cosine', 'cyclic', 'step'
    cyclic_base_lr: float = 1e-5
    cyclic_max_lr: float = 1e-2
    step_lr_gamma: float = 0.1
    step_lr_milestones: List[int] = field(default_factory=lambda: [30, 60, 90])
    
    # Loss Weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    
    # MCTS Parameters
    num_simulations: int = 200
    cpuct: float = 4.0  # Exploration constant (same as original UCT_C)
    temperature: float = 1.0  # Temperature for move selection
    temperature_drop_move: int = 10  # Move to drop temperature
    final_temperature: float = 0.1  # Temperature after drop
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    
    # Self-Play Parameters
    num_self_play_games: int = 100
    max_game_moves: int = 200  # Terminate game if exceeds this
    num_parallel_games: int = 8  # Parallel self-play games for GPU efficiency
    parallel_simulations: int = 1  # Number of parallel MCTS simulations (batched NN inference)
    
    # Progress Display
    show_progress: bool = True  # Show progress bars during training/self-play
    detailed_game_progress_threshold: int = 10  # Show move-by-move progress if num_games <= this
    
    # Evaluation Parameters
    num_evaluation_games: int = 20
    win_threshold: float = 0.55  # 55% win rate to accept new model
    
    # Data Parameters
    replay_buffer_size: int = 100000  # Maximum states in replay buffer
    min_replay_size: int = 10000  # Minimum states before training starts
    augment_data: bool = True  # Use board symmetries for data augmentation
    validation_split: float = 0.2
    
    # System Parameters
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    mixed_precision: bool = True  # AMP for efficient GPU memory usage
    gradient_accumulation_steps: int = 1
    compile_model: bool = False  # torch.compile for PyTorch 2.0+
    
    # Directories
    checkpoint_dir: str = 'data/torch_checkpoints'
    data_dir: str = 'data/torch_training_data'
    log_dir: str = 'data/torch_logs'
    
    # Logging
    log_interval: int = 10  # Log every N batches
    eval_interval: int = 1  # Evaluate every N epochs
    save_interval: int = 1  # Save checkpoint every N epochs
    
    # Random Seed
    seed: Optional[int] = 42
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Ensure device is valid
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = 'cpu'
            self.mixed_precision = False
        
        # Adjust batch size for CPU
        if self.device == 'cpu':
            self.batch_size = min(self.batch_size, 64)
            self.num_parallel_games = min(self.num_parallel_games, 2)
    
    def get_memory_estimate_mb(self) -> float:
        """
        Estimate GPU memory usage in MB.
        
        Returns:
            Estimated memory usage in megabytes.
        """
        # Rough estimates based on model size and batch size
        model_params = self.num_res_blocks * self.num_channels * self.num_channels * 9
        model_memory = model_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        # Activation memory during forward/backward pass
        activation_memory = (
            self.batch_size * self.num_channels * 
            self.board_size * self.board_size * 
            self.num_res_blocks * 4 / (1024 * 1024)
        )
        
        # Mixed precision halves activation memory
        if self.mixed_precision:
            activation_memory /= 2
        
        return model_memory + activation_memory
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            k: getattr(self, k) for k in self.__dataclass_fields__
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create configuration from dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        })


# Pre-defined configurations for different hardware setups
def get_small_config() -> Config:
    """
    Configuration for smaller GPU (8GB VRAM) or CPU training.
    """
    return Config(
        num_res_blocks=6,
        num_channels=64,
        batch_size=128,
        num_simulations=100,
        num_parallel_games=4,
    )


def get_large_config() -> Config:
    """
    Configuration for larger GPU (24GB+ VRAM).
    """
    return Config(
        num_res_blocks=15,
        num_channels=192,
        batch_size=512,
        num_simulations=400,
        num_parallel_games=16,
    )


def get_debug_config() -> Config:
    """
    Configuration for debugging and testing.
    """
    return Config(
        num_res_blocks=2,
        num_channels=32,
        batch_size=16,
        num_simulations=10,
        num_self_play_games=2,
        num_evaluation_games=2,
        num_epochs=2,
        max_game_moves=50,
        show_progress=True,
    )


def get_parallel_config() -> Config:
    """
    Configuration optimized for parallel MCTS simulations.
    Uses batched neural network inference to better utilize GPU VRAM.
    """
    return Config(
        num_res_blocks=10,
        num_channels=128,
        batch_size=256,
        num_simulations=200,
        num_parallel_games=8,
        parallel_simulations=8,  # Batch 8 simulations together
        show_progress=True,
    )
