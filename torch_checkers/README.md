# PyTorch MCTS for Checkers

A GPU-optimized PyTorch implementation of Monte Carlo Tree Search (MCTS) with neural network guidance for playing Checkers. This implementation follows the AlphaZero approach with policy and value heads.

## Overview

This module provides a complete training and evaluation pipeline for learning to play Checkers using:

- **Neural Network**: ResNet-based architecture with dual policy and value heads
- **MCTS**: Monte Carlo Tree Search with PUCT selection and Dirichlet noise exploration
- **Self-Play**: AlphaZero-style training through self-play games
- **Mixed Precision**: Automatic mixed precision (AMP) for efficient GPU memory usage

## Features

### GPU Optimization (16GB VRAM Budget)
- Mixed precision training (FP16) for reduced memory footprint
- Configurable batch sizes optimized for various GPU configurations
- Efficient batched neural network inference during MCTS
- Gradient accumulation for effective larger batch sizes
- Memory-efficient residual blocks with optional SE attention

### Training Pipeline
- Complete AlphaZero-style training loop (self-play → train → evaluate)
- Learning rate scheduling (cosine, cyclic, step)
- Early stopping based on validation loss
- Checkpoint saving and resumption
- Data augmentation through board symmetries

### Evaluation
- Model vs model matches
- Round-robin tournaments
- Statistical analysis of results

## Installation

### Requirements

```bash
pip install torch>=1.12.0
pip install numpy
pip install tabulate
```

Optional (for visualization):
```bash
pip install matplotlib
```

### Quick Start

1. **Train a model from scratch:**
```bash
cd /path/to/Checkers-MCTS
python -m torch_checkers.train --iterations 10 --games 100 --simulations 200
```

2. **Evaluate trained models:**
```bash
python -m torch_checkers.evaluate --model1 data/torch_checkpoints/best_model.pt \
                         --model2 data/torch_checkpoints/checkpoint_iter5.pt \
                         --games 20
```

3. **Play against the AI:**
```bash
python -m torch_checkers.play --model data/torch_checkpoints/best_model.pt --human-first
```

## Architecture

### Neural Network (`model.py`)

The network architecture follows AlphaZero design:

```
Input (14 × 8 × 8)
    ↓
Conv3×3 + BatchNorm + ReLU
    ↓
[Residual Block] × N
    ↓
    ├─→ Policy Head → 512 move probabilities
    │
    └─→ Value Head → scalar in [-1, 1]
```

**Input Channels (from Checkers.py):**
- Channels 0-1: Player 1's men and kings
- Channels 2-3: Player 2's men and kings
- Channel 4: Current player indicator
- Channel 5: Draw counter
- Channels 6-13: Legal move indicators (4 directions × 2 types)

**Residual Block:**
```
Input → Conv3×3 → BN → ReLU → Conv3×3 → BN → Add(Input) → ReLU
```

**Optional SE (Squeeze-and-Excitation) Attention:**
```
Input → GlobalAvgPool → FC → ReLU → FC → Sigmoid → Scale(Input)
```

### MCTS (`mcts.py`)

The MCTS implementation uses the PUCT (Predictor + UCT) selection criterion:

```
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
```

Where:
- `Q(s,a)`: Mean value of action `a` from state `s`
- `P(s,a)`: Prior probability from neural network
- `N(s)`: Visit count of parent state
- `N(s,a)`: Visit count of action
- `c_puct`: Exploration constant (default: 4.0)

**Features:**
- Dirichlet noise at root for exploration during training
- Temperature-based move selection
- Efficient tree reuse between moves
- Terminal state handling

### Configuration (`config.py`)

All hyperparameters are centralized in the `Config` dataclass:

```python
from torch_checkers.config import Config

# Default configuration (16GB VRAM)
config = Config(
    num_res_blocks=10,
    num_channels=128,
    batch_size=256,
    num_simulations=200,
)

# Small configuration (8GB VRAM)
from torch_checkers.config import get_small_config
config = get_small_config()

# Debug configuration
from torch_checkers.config import get_debug_config
config = get_debug_config()
```

## Usage

### Training

**Basic Training:**
```bash
python -m torch_checkers.train --iterations 10 --games 100 --epochs 10
```

**Full Options:**
```bash
python -m torch_checkers.train \
    --iterations 10 \        # Training iterations
    --games 100 \            # Self-play games per iteration
    --simulations 200 \      # MCTS simulations per move
    --epochs 10 \            # Training epochs per iteration
    --batch-size 256 \       # Training batch size
    --lr 0.001 \             # Learning rate
    --res-blocks 10 \        # Number of residual blocks
    --channels 128 \         # Hidden channels
    --device cuda \          # Device (cuda/cpu)
    --checkpoint-dir data/torch_checkpoints
```

**Resume from checkpoint:**
```bash
python -m torch_checkers.train --checkpoint data/torch_checkpoints/checkpoint_iter5.pt
```

### Evaluation

**Match between two models:**
```bash
python -m torch_checkers.evaluate \
    --model1 path/to/model1.pt \
    --model2 path/to/model2.pt \
    --games 20
```

**Tournament between multiple models:**
```bash
python -m torch_checkers.evaluate \
    --tournament model1.pt model2.pt model3.pt \
    --games 10
```

### Playing

**Play against AI (human moves second):**
```bash
python -m torch_checkers.play --model data/torch_checkpoints/best_model.pt
```

**Play as first player:**
```bash
python -m torch_checkers.play --model data/torch_checkpoints/best_model.pt --human-first
```

**Watch AI vs AI:**
```bash
python -m torch_checkers.play --model data/torch_checkpoints/best_model.pt --ai-vs-ai --delay 2
```

## Module Reference

### `config.py`
Configuration management with `Config` dataclass and preset configurations.

### `model.py`
Neural network architecture including:
- `CheckersModel`: Main network with policy and value heads
- `ResidualBlock`: Standard residual block
- `ResidualBlockSE`: Residual block with SE attention
- `PolicyHead`: Policy output head (512 moves)
- `ValueHead`: Value output head (scalar)

### `mcts.py`
Monte Carlo Tree Search implementation:
- `MCTSNode`: Tree node representation
- `MCTSPlayer`: MCTS controller with neural network integration
- `run_self_play_game()`: Function to generate training data

### `dataset.py`
Data handling utilities:
- `CheckersDataset`: PyTorch Dataset for training data
- `ReplayBuffer`: Experience replay buffer for self-play
- `create_data_loaders()`: Factory function for DataLoaders

### `trainer.py`
Training loop implementation:
- `Trainer`: Main training class with mixed precision support
- Learning rate scheduling
- Early stopping
- Checkpoint management

### `utils.py`
Utility functions:
- Logging setup
- Checkpoint save/load
- Training metrics tracking
- Visualization helpers

### `train.py`
Main training script with complete AlphaZero pipeline.

### `evaluate.py`
Model evaluation and tournament script.

### `play.py`
Interactive gameplay script.

## Memory Estimation

Approximate GPU memory usage for default configuration:

| Component | Memory (MB) |
|-----------|-------------|
| Model parameters | ~50 |
| Forward pass activations | ~500 |
| Backward pass gradients | ~500 |
| Optimizer states | ~100 |
| Batch data | ~100 |
| **Total (FP32)** | **~1,250** |
| **Total (FP16)** | **~750** |

With default batch size of 256 and mixed precision, this implementation comfortably fits within a 16GB VRAM budget, leaving room for MCTS inference during self-play.

## Training Tips

1. **Start with debug config** to verify setup:
   ```bash
   python -m torch_checkers.train --debug
   ```

2. **Increase simulations** for stronger play at the cost of speed:
   ```bash
   python -m torch_checkers.train --simulations 400
   ```

3. **Use more games** for diverse training data:
   ```bash
   python -m torch_checkers.train --games 500
   ```

4. **Adjust batch size** based on your GPU:
   - 8GB VRAM: `--batch-size 128`
   - 16GB VRAM: `--batch-size 256`
   - 24GB+ VRAM: `--batch-size 512`

5. **Monitor training** with TensorBoard or log files in `data/torch_logs/`

## File Structure

```
torch_checkers/
├── __init__.py      # Module exports
├── config.py        # Configuration dataclass
├── model.py         # Neural network architecture
├── mcts.py          # Monte Carlo Tree Search
├── dataset.py       # Data loading utilities
├── trainer.py       # Training loop
├── utils.py         # Utility functions
├── train.py         # Main training script
├── evaluate.py      # Evaluation script
├── play.py          # Interactive play script
└── README.md        # This file
```

## Comparison with Original Implementation

| Feature | Original (Keras/TF) | This (PyTorch) |
|---------|---------------------|----------------|
| Framework | Keras/TensorFlow | PyTorch |
| Mixed Precision | No | Yes (AMP) |
| Gradient Checkpointing | No | Optional |
| Configuration | Scattered kwargs | Centralized dataclass |
| MCTS Integration | Separate | Tightly integrated |
| CLI Interface | Script variables | argparse |
| Logging | Basic print | Python logging |
| Data Augmentation | No | Yes (symmetries) |

## License

Same as the parent Checkers-MCTS repository.

## Acknowledgments

This implementation is based on:
- Original Checkers-MCTS repository game logic
- AlphaZero paper by DeepMind
- PyTorch best practices for efficient training
