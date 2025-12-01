"""
###############################################################################
# torch/model.py
#
# PyTorch neural network model for Checkers MCTS.
#
# This module implements a ResNet-based neural network with dual heads:
# - Policy head: Predicts probability distribution over legal moves
# - Value head: Predicts expected game outcome from current position
#
# Architecture follows AlphaZero/AlphaGo Zero design with modifications
# optimized for Checkers' 8x8 board and action space.
#
# GPU Optimization:
# - Supports mixed precision (float16) for efficient memory usage
# - BatchNorm layers for stable training
# - Efficient residual connections
# - Memory-efficient attention optional for larger models
#
###############################################################################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers.
    
    Architecture:
        Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add(Input) -> ReLU
    
    Args:
        num_channels: Number of input and output channels.
        
    Note:
        Uses pre-activation style for better gradient flow with deep networks.
    """
    
    def __init__(self, num_channels: int):
        """Initialize residual block layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channels, num_channels, 
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            
        Returns:
            Output tensor with same shape as input.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.relu(out, inplace=True)
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    This block adaptively recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels.
    
    Args:
        channels: Number of channels in input/output.
        reduction: Reduction ratio for bottleneck.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """Initialize SE block layers."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SE block.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            
        Returns:
            Channel-reweighted tensor with same shape as input.
        """
        batch, channels, _, _ = x.size()
        # Squeeze
        y = self.pool(x).view(batch, channels)
        # Excitation
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y))
        # Scale
        y = y.view(batch, channels, 1, 1)
        return x * y


class ResidualBlockSE(nn.Module):
    """
    Residual block with Squeeze-and-Excitation attention.
    
    Combines residual learning with channel attention for
    improved feature representation.
    
    Args:
        num_channels: Number of input/output channels.
        se_reduction: SE block reduction ratio.
    """
    
    def __init__(self, num_channels: int, se_reduction: int = 16):
        """Initialize residual SE block layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channels, num_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.se = SEBlock(num_channels, se_reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SE attention."""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + residual
        out = F.relu(out, inplace=True)
        return out


class PolicyHead(nn.Module):
    """
    Policy head for predicting move probabilities.
    
    Architecture:
        Input -> Conv3x3 -> BN -> ReLU -> Conv1x1(8ch) -> BN -> Flatten -> Dense(512)
    
    The output is a 512-dimensional probability distribution over possible moves:
    - 8 move directions (4 normal + 4 jumps) x 64 positions = 512
    
    Args:
        input_channels: Number of input channels from backbone.
        board_size: Size of the board (8 for Checkers).
        output_size: Number of possible moves (512).
    """
    
    def __init__(
        self, 
        input_channels: int,
        board_size: int = 8,
        output_size: int = 512
    ):
        """Initialize policy head layers."""
        super().__init__()
        self.board_size = board_size
        self.output_size = output_size
        
        # First conv to process features
        self.conv1 = nn.Conv2d(
            input_channels, input_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(input_channels)
        
        # Reduce to 8 channels (one per move direction)
        self.conv2 = nn.Conv2d(
            input_channels, 8,
            kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(8)
        
        # Output layer
        self.fc = nn.Linear(8 * board_size * board_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy head.
        
        Args:
            x: Input tensor from backbone, shape (batch, channels, 8, 8).
            
        Returns:
            Policy logits of shape (batch, 512). Apply softmax for probabilities.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ValueHead(nn.Module):
    """
    Value head for predicting game outcome.
    
    Architecture:
        Input -> Conv1x1 -> BN -> ReLU -> Flatten -> Dense(256) -> ReLU -> Dense(1) -> Tanh
    
    The output is a scalar in [-1, 1]:
    - +1: Current player wins
    -  0: Draw
    - -1: Current player loses
    
    Args:
        input_channels: Number of input channels from backbone.
        board_size: Size of the board (8 for Checkers).
        hidden_size: Size of hidden dense layer.
    """
    
    def __init__(
        self,
        input_channels: int,
        board_size: int = 8,
        hidden_size: int = 256
    ):
        """Initialize value head layers."""
        super().__init__()
        self.board_size = board_size
        
        # Reduce channels
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        
        # Dense layers
        self.fc1 = nn.Linear(board_size * board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.
        
        Args:
            x: Input tensor from backbone, shape (batch, channels, 8, 8).
            
        Returns:
            Value prediction of shape (batch, 1) in range [-1, 1].
        """
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        out = torch.tanh(out)
        return out


class CheckersModel(nn.Module):
    """
    Complete neural network for Checkers with policy and value heads.
    
    This model takes the Checkers game state as input and outputs:
    1. Policy: Probability distribution over all possible moves
    2. Value: Expected game outcome for current player
    
    The architecture follows AlphaZero design:
    - Input convolution to project state features to hidden channels
    - Stack of residual blocks for feature extraction
    - Separate policy and value heads for dual predictions
    
    Input State Format (from Checkers.py):
        The input is a 14-channel 8x8 tensor:
        - Channel 0: Player 1's men positions
        - Channel 1: Player 1's kings positions
        - Channel 2: Player 2's men positions
        - Channel 3: Player 2's kings positions
        - Channel 4: Current player indicator (0s or 1s)
        - Channel 5: Draw counter (normalized)
        - Channels 6-9: Normal moves (UL, UR, BL, BR)
        - Channels 10-13: Jump moves (UL, UR, BL, BR)
    
    Args:
        config: Configuration object with model hyperparameters.
        use_se: Whether to use Squeeze-and-Excitation blocks.
    
    Example:
        >>> config = Config(num_res_blocks=10, num_channels=128)
        >>> model = CheckersModel(config)
        >>> state = torch.randn(32, 14, 8, 8)  # Batch of 32 states
        >>> policy, value = model(state)
        >>> policy.shape  # (32, 512)
        >>> value.shape   # (32, 1)
    """
    
    def __init__(self, config, use_se: bool = False):
        """
        Initialize the Checkers neural network.
        
        Args:
            config: Configuration object containing:
                - input_channels: Number of input feature planes (14)
                - num_channels: Hidden layer channels (128)
                - num_res_blocks: Number of residual blocks (10)
                - board_size: Board size (8)
                - policy_output_size: Policy output dimension (512)
            use_se: Use SE attention in residual blocks.
        """
        super().__init__()
        self.config = config
        
        # Input convolution
        self.input_conv = nn.Conv2d(
            config.input_channels,
            config.num_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(config.num_channels)
        
        # Residual tower
        block_class = ResidualBlockSE if use_se else ResidualBlock
        self.res_blocks = nn.ModuleList([
            block_class(config.num_channels)
            for _ in range(config.num_res_blocks)
        ])
        
        # Policy and value heads
        self.policy_head = PolicyHead(
            config.num_channels,
            config.board_size,
            config.policy_output_size
        )
        self.value_head = ValueHead(
            config.num_channels,
            config.board_size
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights for stable training.
        
        Uses Kaiming initialization for conv layers and
        Xavier initialization for linear layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch, 14, 8, 8).
               Expected to be in channels-first format.
        
        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Shape (batch, 512), raw logits for moves
                - value: Shape (batch, 1), value prediction in [-1, 1]
        """
        # Input projection
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out, inplace=True)
        
        # Residual tower
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Dual heads
        policy = self.policy_head(out)
        value = self.value_head(out)
        
        return policy, value
    
    def predict(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with optional action masking.
        
        This method is used during MCTS for inference. It applies
        softmax to policy outputs and optionally masks illegal moves.
        
        Args:
            state: Game state tensor, shape (batch, 14, 8, 8).
            action_mask: Binary mask for legal moves, shape (batch, 512).
                        1 for legal moves, 0 for illegal.
        
        Returns:
            Tuple of (policy_probs, value):
                - policy_probs: Normalized probabilities, shape (batch, 512)
                - value: Value prediction, shape (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(state)
            
            if action_mask is not None:
                # Mask illegal moves with large negative value
                policy_logits = policy_logits.masked_fill(
                    action_mask == 0, -1e9
                )
            
            policy_probs = F.softmax(policy_logits, dim=1)
            
        return policy_probs, value
    
    @torch.jit.export
    def get_policy_value(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TorchScript-compatible forward pass.
        
        This method is exported for use with TorchScript compilation.
        
        Args:
            x: Input tensor, shape (batch, 14, 8, 8).
            
        Returns:
            Tuple of (policy_probs, value).
        """
        policy_logits, value = self.forward(x)
        policy_probs = F.softmax(policy_logits, dim=1)
        return policy_probs, value
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_footprint(self) -> float:
        """Return model memory footprint in MB."""
        param_size = sum(
            p.nelement() * p.element_size() for p in self.parameters()
        )
        buffer_size = sum(
            b.nelement() * b.element_size() for b in self.buffers()
        )
        return (param_size + buffer_size) / (1024 * 1024)


def create_model(config, compile_model: bool = False) -> CheckersModel:
    """
    Factory function to create and optionally compile the model.
    
    Args:
        config: Configuration object.
        compile_model: Whether to use torch.compile (PyTorch 2.0+).
        
    Returns:
        CheckersModel instance, optionally compiled.
    """
    model = CheckersModel(config)
    model = model.to(config.device)
    
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
    
    return model


def convert_state_for_model(state, device: str = 'cuda') -> torch.Tensor:
    """
    Convert Checkers.py state format to model input format.
    
    The original Checkers state is a (15, 8, 8) numpy array.
    We use only the first 14 channels as model input.
    
    Args:
        state: NumPy array of shape (15, 8, 8) from Checkers.py.
        device: Target device for tensor.
        
    Returns:
        Tensor of shape (1, 14, 8, 8) ready for model input.
    """
    import numpy as np
    
    if isinstance(state, np.ndarray):
        # Take first 14 channels (exclude channel 14 which has action indices)
        state_tensor = torch.from_numpy(state[:14].copy()).float()
    else:
        state_tensor = state[:14].float()
    
    # Add batch dimension if needed
    if state_tensor.dim() == 3:
        state_tensor = state_tensor.unsqueeze(0)
    
    return state_tensor.to(device)


def get_action_mask_from_state(state) -> torch.Tensor:
    """
    Extract action mask from Checkers state.
    
    The action mask indicates which moves are legal based on
    channels 6-13 of the state (move indicator planes).
    
    Args:
        state: NumPy array of shape (15, 8, 8) or tensor.
        
    Returns:
        Binary mask tensor of shape (512,) or (batch, 512).
    """
    import numpy as np
    
    if isinstance(state, np.ndarray):
        action_planes = state[6:14]
        mask = torch.from_numpy(action_planes.flatten()).float()
    else:
        if state.dim() == 3:
            action_planes = state[6:14]
            mask = action_planes.reshape(-1)
        else:  # Batched
            action_planes = state[:, 6:14]
            mask = action_planes.reshape(state.size(0), -1)
    
    return (mask > 0).float()
