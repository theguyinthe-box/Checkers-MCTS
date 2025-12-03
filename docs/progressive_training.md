# Progressive Training Experiment

## Overview

The `experiment_progressive.py` script implements a training pipeline that demonstrates **progressive improvement** of a Checkers AI model over multiple training iterations. Unlike the original experiment pipeline which had issues with learning rate scheduler resets, this implementation:

1. **Maintains cumulative training data** - All self-play data is retained and used for training
2. **Preserves optimizer state** - Learning rate and momentum are maintained across iterations
3. **Uses fixed learning rate** - Avoids scheduler resets between iterations for stable learning
4. **Tracks global epochs** - Properly counts total training epochs across all iterations
5. **Evaluates against random player** - Validates improvement through periodic testing

## Quick Start

### Debug Mode (Fast Testing)
```bash
# Quick test run with minimal settings
python -m torch_checkers.experiment_progressive --debug --iterations 5

# With more evaluation games for better statistics
python -m torch_checkers.experiment_progressive --debug --iterations 5 --eval-games 10
```

### Full Training Run
```bash
python -m torch_checkers.experiment_progressive \
    --iterations 20 \
    --games-per-iteration 50 \
    --simulations 100 \
    --eval-games 20 \
    --epochs 5
```

## Command Line Arguments

### Training Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--iterations` | 10 | Number of training iterations |
| `--games-per-iteration` | 50 | Self-play games per iteration |
| `--simulations` | 100 | MCTS simulations per move during training |
| `--epochs` | 5 | Training epochs per iteration |
| `--max-moves` | 200 | Maximum moves per game |

### Model Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--res-blocks` | 6 | Number of residual blocks |
| `--channels` | 64 | Channels in convolutional layers |
| `--batch-size` | 128 | Training batch size |
| `--lr` | 1e-3 | Learning rate (fixed) |

### Evaluation Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--eval-games` | 20 | Evaluation games vs random player |
| `--eval-interval` | 1 | Evaluate every N iterations |
| `--eval-simulations` | 50 | MCTS simulations during evaluation |

### System Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | auto | cuda if available, else cpu |
| `--seed` | 42 | Random seed |
| `--workers` | 0 | Data loading workers |
| `--output-dir` | data/progressive_experiment | Output directory |
| `--checkpoint` | None | Resume from checkpoint |
| `--no-progress` | False | Disable progress bars |
| `--debug` | False | Use debug configuration |

## Training Pipeline

Each iteration consists of three phases:

### Phase 1: Self-Play Data Generation
- Play games using MCTS with the current model
- Collect (state, policy, value) tuples for training
- Data is added to a cumulative replay buffer

### Phase 2: Training
- Train on ALL accumulated data (not just current iteration)
- Uses fixed learning rate for stability across iterations
- Optimizer momentum is preserved between training calls
- Global epoch counter tracks total epochs trained

### Phase 3: Evaluation
- Play evaluation games against random player
- Track win rate, wins, losses, and draws
- Log improvement/regression from previous iteration

## Output Files

The experiment generates several outputs:

```
data/progressive_experiment/
├── checkpoints/
│   ├── checkpoint_iter1_TIMESTAMP.pt
│   ├── checkpoint_iter2_TIMESTAMP.pt
│   └── final_model_TIMESTAMP.pt
├── data/
│   └── (replay buffer data)
├── logs/
│   ├── training_TIMESTAMP.log
│   └── progressive_results_TIMESTAMP.png
└── experiment_results_TIMESTAMP.json
```

### Result Files
- **checkpoints/**: Model weights at each iteration and final model
- **logs/**: Training logs and visualization plots
- **experiment_results.json**: Complete experiment results in JSON format

## Expected Results

With sufficient training:
- Initial win rate vs random: ~50% (untrained model)
- After training: Should increase to >70-80%
- Loss should decrease consistently across epochs

## Example Output

```
======================================================================
EXPERIMENT SUMMARY
======================================================================

Training iterations: 20
Total epochs trained: 100
Total examples in replay buffer: 15000

Final win rate vs random: 85.0%

Progressive Improvement:
  Initial win rate: 50.0%
  Final win rate:   85.0%
  Improvement:      +35.0%

✓ SUCCESS: Model showed progressive improvement during training!

Evaluation History:
  Iteration  Win Rate    Model Wins    Random Wins    Draws
-----------  --------    ----------    -----------    -----
          0  50.0%              10             10        0
          5  65.0%              13              7        0
         10  75.0%              15              5        0
         15  80.0%              16              4        0
         20  85.0%              17              3        0
```

## Troubleshooting

### Model Not Improving
- Increase `--games-per-iteration` to generate more training data
- Increase `--epochs` for more thorough training per iteration
- Increase `--simulations` for better quality self-play data
- Try decreasing `--lr` for more stable learning

### Training Too Slow
- Decrease `--simulations` (faster but lower quality data)
- Use `--no-progress` to reduce output overhead
- Use GPU with `--device cuda`

### Out of Memory
- Decrease `--batch-size`
- Decrease `--channels` or `--res-blocks`

## Differences from Original Pipeline

The original `experiment_single.py` had issues with:
1. Learning rate scheduler reset every iteration
2. Missing output directory creation
3. Trainer recreated each iteration (losing state)

This progressive experiment fixes these by:
1. Using fixed learning rate (`lr_schedule='constant'`)
2. Creating all directories upfront
3. Maintaining a single `ProgressiveTrainer` instance
4. Properly tracking global epochs across all iterations
