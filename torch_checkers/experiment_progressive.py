#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# torch_checkers/experiment_progressive.py
#
# Progressive Training Experiment: Single Player Training with Cumulative Learning
#
# This script implements a training pipeline that demonstrates progressive
# improvement of a Checkers AI model over multiple training iterations.
#
# Key Features:
# - Cumulative training data: All self-play data is retained and used for training
# - Persistent optimizer state: Learning rate and momentum are maintained across iterations
# - Fixed learning rate: Avoids scheduler resets between iterations for stable learning
# - Regular evaluation: Model is tested against random player to track improvement
# - Comprehensive logging: Tracks win rate, loss, and other metrics over time
#
# The model should progressively get better (more competitive) as training
# progresses, which is validated through periodic evaluation against a
# random player baseline.
#
# Usage:
#   # Quick test with debug settings
#   python -m torch_checkers.experiment_progressive --debug --iterations 5
#
#   # Full training run
#   python -m torch_checkers.experiment_progressive \
#       --iterations 20 \
#       --games-per-iteration 50 \
#       --simulations 100 \
#       --eval-games 20 \
#       --epochs 5
#
# Expected Outcome:
#   The model's win rate against the random player should increase from
#   ~50% initially to >80% after sufficient training iterations.
#
###############################################################################
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_checkers.config import Config, get_debug_config
from torch_checkers.model import CheckersModel, create_model
from torch_checkers.mcts import MCTSPlayer
from torch_checkers.dataset import ReplayBuffer, create_data_loaders
from torch_checkers.random_player import RandomPlayer
from torch_checkers.train import run_self_play
from torch_checkers.utils import (
    setup_logging,
    set_seed,
    save_checkpoint,
    create_timestamp,
    get_gpu_memory_info,
    AverageMeter,
)

from Checkers import Checkers

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers=None, tablefmt=None, showindex=None):
        """Simple fallback tabulate that formats data as tab-separated values."""
        # Note: tablefmt and showindex are accepted for interface compatibility but not used
        lines = []
        if headers:
            lines.append('\t'.join(str(h) for h in headers))
        for row in data:
            lines.append('\t'.join(str(c) for c in row))
        return '\n'.join(lines)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Progressive Training: Train a Checkers AI with cumulative improvement'
    )
    
    # Training parameters
    parser.add_argument(
        '--iterations', type=int, default=10,
        help='Number of training iterations (default: 10)'
    )
    parser.add_argument(
        '--games-per-iteration', type=int, default=50,
        help='Number of self-play games per iteration (default: 50)'
    )
    parser.add_argument(
        '--simulations', type=int, default=100,
        help='MCTS simulations per move during training (default: 100)'
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help='Training epochs per iteration (default: 5)'
    )
    parser.add_argument(
        '--max-moves', type=int, default=200,
        help='Maximum moves per game (default: 200)'
    )
    
    # Model parameters
    parser.add_argument(
        '--res-blocks', type=int, default=6,
        help='Number of residual blocks (default: 6)'
    )
    parser.add_argument(
        '--channels', type=int, default=64,
        help='Number of channels in convolutions (default: 64)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Training batch size (default: 128)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate (default: 1e-3). Uses fixed LR to avoid scheduler resets.'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--eval-games', type=int, default=20,
        help='Number of evaluation games against random player (default: 20)'
    )
    parser.add_argument(
        '--eval-interval', type=int, default=1,
        help='Evaluate every N iterations (default: 1)'
    )
    parser.add_argument(
        '--eval-simulations', type=int, default=50,
        help='MCTS simulations per move during evaluation (default: 50)'
    )
    
    # System parameters
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for training (default: cuda if available)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--workers', type=int, default=0,
        help='Number of data loading workers (default: 0 for main thread)'
    )
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/progressive_experiment',
        help='Directory for outputs (default: data/progressive_experiment)'
    )
    
    # Debug mode
    parser.add_argument(
        '--debug', action='store_true',
        help='Use debug configuration (small model, few games)'
    )
    
    # Progress display
    parser.add_argument(
        '--no-progress', action='store_true',
        help='Disable progress bars'
    )
    
    return parser.parse_args()


def create_config(args) -> Config:
    """Create configuration from command line arguments."""
    if args.debug:
        config = get_debug_config()
        # Override some debug settings for better demonstration
        config.num_simulations = 20
        config.num_self_play_games = 5
        config.num_epochs = 3
        config.max_game_moves = 100
    else:
        config = Config(
            # Model
            num_res_blocks=args.res_blocks,
            num_channels=args.channels,
            
            # Training - Use fixed LR schedule to avoid resets
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            lr_schedule='constant',  # Use constant LR to avoid scheduler issues
            
            # MCTS
            num_simulations=args.simulations,
            max_game_moves=args.max_moves,
            
            # Self-play
            num_self_play_games=args.games_per_iteration,
            
            # System
            device=args.device,
            num_workers=args.workers,
            seed=args.seed,
        )
    
    # Set directories
    config.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    config.data_dir = os.path.join(args.output_dir, 'data')
    config.log_dir = os.path.join(args.output_dir, 'logs')
    
    config.show_progress = not args.no_progress
    return config


class ProgressiveTrainer:
    """
    Trainer that maintains state across training iterations for cumulative learning.
    
    Unlike the standard Trainer which may reset state, this class:
    - Maintains optimizer momentum across iterations
    - Uses a fixed learning rate to avoid scheduler resets
    - Properly tracks global epoch count
    """
    
    def __init__(self, model: CheckersModel, config: Config):
        """Initialize the progressive trainer."""
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Setup optimizer with fixed learning rate
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training state - preserved across iterations
        self.global_epoch = 0
        self.global_step = 0
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'policy_loss': [],
            'value_loss': [],
        }
        
        # Loss weights
        self.policy_weight = config.policy_loss_weight
        self.value_weight = config.value_loss_weight
        
        # Logging
        self.logger = setup_logging(config.log_dir)
    
    def train_iteration(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray, float, float]],
        num_epochs: int
    ) -> Dict[str, List[float]]:
        """
        Train for one iteration (multiple epochs) on the accumulated data.
        
        Args:
            training_data: All accumulated training examples.
            num_epochs: Number of epochs for this iteration.
            
        Returns:
            Dictionary with training metrics for this iteration.
        """
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            training_data,
            val_split=self.config.validation_split,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            augment=self.config.augment_data
        )
        
        iteration_history = {
            'train_loss': [],
            'val_loss': [],
            'policy_loss': [],
            'value_loss': [],
        }
        
        show_progress = getattr(self.config, 'show_progress', True)
        
        for epoch in range(num_epochs):
            self.global_epoch += 1
            
            # Training phase
            self.model.train()
            loss_meter = AverageMeter()
            policy_loss_meter = AverageMeter()
            value_loss_meter = AverageMeter()
            
            epoch_pbar = tqdm(
                train_loader,
                desc=f"  Epoch {self.global_epoch}",
                disable=not show_progress,
                leave=False
            )
            
            for batch in epoch_pbar:
                states = batch['state'].to(self.device)
                target_policies = batch['policy'].to(self.device)
                target_values = batch['value'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                policy_logits, value_preds = self.model(states)
                
                # Losses
                policy_loss = F.cross_entropy(policy_logits, target_policies)
                value_loss = F.mse_loss(value_preds, target_values)
                loss = self.policy_weight * policy_loss + self.value_weight * value_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.global_step += 1
                
                # Update meters
                batch_size = states.size(0)
                loss_meter.update(loss.item(), batch_size)
                policy_loss_meter.update(policy_loss.item(), batch_size)
                value_loss_meter.update(value_loss.item(), batch_size)
                
                epoch_pbar.set_postfix({
                    'loss': f"{loss_meter.avg:.4f}",
                    'p_loss': f"{policy_loss_meter.avg:.4f}",
                    'v_loss': f"{value_loss_meter.avg:.4f}"
                })
            
            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                val_loss_meter = AverageMeter()
                with torch.no_grad():
                    for batch in val_loader:
                        states = batch['state'].to(self.device)
                        target_policies = batch['policy'].to(self.device)
                        target_values = batch['value'].to(self.device)
                        
                        policy_logits, value_preds = self.model(states)
                        policy_loss = F.cross_entropy(policy_logits, target_policies)
                        value_loss = F.mse_loss(value_preds, target_values)
                        loss = self.policy_weight * policy_loss + self.value_weight * value_loss
                        
                        val_loss_meter.update(loss.item(), states.size(0))
                
                val_loss = val_loss_meter.avg
            
            # Record history
            iteration_history['train_loss'].append(loss_meter.avg)
            iteration_history['val_loss'].append(val_loss)
            iteration_history['policy_loss'].append(policy_loss_meter.avg)
            iteration_history['value_loss'].append(value_loss_meter.avg)
            
            self.training_history['epochs'].append(self.global_epoch)
            self.training_history['train_loss'].append(loss_meter.avg)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['policy_loss'].append(policy_loss_meter.avg)
            self.training_history['value_loss'].append(value_loss_meter.avg)
            
            if not show_progress:
                self.logger.info(
                    f"Epoch {self.global_epoch}: train_loss={loss_meter.avg:.4f}, "
                    f"val_loss={val_loss:.4f}, policy_loss={policy_loss_meter.avg:.4f}, "
                    f"value_loss={value_loss_meter.avg:.4f}"
                )
        
        return iteration_history
    
    def get_training_summary(self) -> Dict:
        """Get summary of all training so far."""
        return {
            'global_epoch': self.global_epoch,
            'global_step': self.global_step,
            'history': self.training_history,
        }


def evaluate_against_random(
    model: CheckersModel,
    config: Config,
    num_games: int,
    logger
) -> Dict:
    """
    Evaluate the trained model against a random player.
    
    Returns evaluation results dictionary.
    """
    model.eval()
    
    results = {
        'model_wins': 0,
        'random_wins': 0,
        'draws': 0,
        'games': [],
        'total_moves': 0
    }
    
    show_progress = getattr(config, 'show_progress', True)
    games_pbar = tqdm(
        range(num_games),
        desc="  Evaluation",
        disable=not show_progress,
        unit="game",
        leave=False
    )
    
    for game_num in games_pbar:
        game_env = Checkers()
        
        # Alternate who plays first
        model_is_p1 = (game_num % 2 == 0)
        mcts_player = MCTSPlayer(game_env, model, config, config.device)
        random_player = RandomPlayer(game_env)
        
        move_count = 0
        while not game_env.done and move_count < config.max_game_moves:
            current_player = int(game_env.state[4, 0, 0])
            
            if (current_player == 0) == model_is_p1:
                # Model's turn
                next_state, _, _ = mcts_player.get_action_probs(
                    game_env.state,
                    game_env.history,
                    temperature=0,
                    add_noise=False
                )
            else:
                # Random player's turn
                next_state, _, _ = random_player.get_action(
                    game_env.state,
                    game_env.history
                )
            
            game_env.step(next_state)
            move_count += 1
        
        results['total_moves'] += move_count
        
        # Determine winner
        if not game_env.done:
            state = game_env.state
            p1_count = np.sum(state[0:2])
            p2_count = np.sum(state[2:4])
            if p1_count > p2_count:
                outcome = 'player1_wins'
            elif p2_count > p1_count:
                outcome = 'player2_wins'
            else:
                outcome = 'draw'
        else:
            outcome = game_env.outcome
        
        # Record result
        if outcome == 'draw':
            results['draws'] += 1
        elif outcome == 'player1_wins':
            if model_is_p1:
                results['model_wins'] += 1
            else:
                results['random_wins'] += 1
        else:
            if not model_is_p1:
                results['model_wins'] += 1
            else:
                results['random_wins'] += 1
        
        results['games'].append({
            'game_num': game_num + 1,
            'model_first': model_is_p1,
            'outcome': outcome,
            'moves': move_count
        })
        
        games_pbar.set_postfix({
            'Model': results['model_wins'],
            'Random': results['random_wins'],
            'Draw': results['draws']
        })
    
    # Calculate statistics
    total_games = results['model_wins'] + results['random_wins'] + results['draws']
    results['win_rate'] = (results['model_wins'] + 0.5 * results['draws']) / total_games
    results['avg_moves'] = results['total_moves'] / total_games
    
    return results


def plot_progressive_results(
    eval_history: List[Dict],
    training_summary: Dict,
    save_dir: str
):
    """Plot the progressive improvement of the model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Win rate over iterations
    iterations = [h['iteration'] for h in eval_history]
    win_rates = [h['win_rate'] * 100 for h in eval_history]
    
    axes[0, 0].plot(iterations, win_rates, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=50, color='r', linestyle='--', label='Random baseline (50%)')
    axes[0, 0].set_xlabel('Training Iteration')
    axes[0, 0].set_ylabel('Win Rate vs Random (%)')
    axes[0, 0].set_title('Model Win Rate Over Training')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Wins breakdown
    model_wins = [h['model_wins'] for h in eval_history]
    random_wins = [h['random_wins'] for h in eval_history]
    draws = [h['draws'] for h in eval_history]
    
    x = np.arange(len(iterations))
    width = 0.25
    axes[0, 1].bar(x - width, model_wins, width, label='Model Wins', color='green')
    axes[0, 1].bar(x, random_wins, width, label='Random Wins', color='red')
    axes[0, 1].bar(x + width, draws, width, label='Draws', color='gray')
    axes[0, 1].set_xlabel('Training Iteration')
    axes[0, 1].set_ylabel('Number of Games')
    axes[0, 1].set_title('Game Outcomes by Iteration')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(iterations)
    axes[0, 1].legend()
    axes[0, 1].grid(True, axis='y')
    
    # Plot 3: Training loss over epochs
    history = training_summary['history']
    if history['epochs']:
        epochs = history['epochs']
        axes[1, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        if any(v > 0 for v in history['val_loss']):
            axes[1, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[1, 0].set_xlabel('Global Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot 4: Policy and Value loss
    if history['epochs']:
        axes[1, 1].plot(epochs, history['policy_loss'], 'g-', label='Policy Loss')
        axes[1, 1].plot(epochs, history['value_loss'], 'm-', label='Value Loss')
        axes[1, 1].set_xlabel('Global Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Policy and Value Loss Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'progressive_results_{create_timestamp()}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    return plot_path


def main():
    """Main experiment function."""
    args = parse_args()
    config = create_config(args)
    
    # Create all necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config.log_dir)
    
    # Set seed for reproducibility
    if config.seed is not None:
        set_seed(config.seed)
    
    logger.info("=" * 70)
    logger.info("PROGRESSIVE TRAINING EXPERIMENT")
    logger.info("=" * 70)
    logger.info("This experiment trains a Checkers AI model progressively,")
    logger.info("demonstrating improvement over multiple training iterations.")
    logger.info("-" * 70)
    logger.info(f"Training iterations: {args.iterations}")
    logger.info(f"Self-play games per iteration: {config.num_self_play_games}")
    logger.info(f"MCTS simulations per move: {config.num_simulations}")
    logger.info(f"Training epochs per iteration: {config.num_epochs}")
    logger.info(f"Evaluation games: {args.eval_games}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Learning rate: {config.learning_rate} (fixed)")
    
    if config.device == 'cuda':
        mem_info = get_gpu_memory_info()
        logger.info(f"GPU Memory: {mem_info['total']:.1f} GB total")
    
    # Create model
    logger.info("-" * 70)
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Model memory: {model.get_memory_footprint():.1f} MB")
    
    # Load checkpoint if provided
    start_iteration = 0
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_iteration = checkpoint.get('iteration', 0) + 1
        logger.info(f"Resuming from iteration {start_iteration}")
    
    # Initialize replay buffer for cumulative data
    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)
    
    # Initialize progressive trainer (maintains state across iterations)
    trainer = ProgressiveTrainer(model, config)
    
    # Evaluation config
    eval_config = Config.from_dict(config.to_dict())
    eval_config.num_simulations = args.eval_simulations
    
    # Evaluation history
    eval_history = []
    
    # Initial evaluation (iteration 0)
    logger.info("-" * 70)
    logger.info("INITIAL EVALUATION (before training)")
    logger.info("-" * 70)
    
    initial_eval = evaluate_against_random(model, eval_config, args.eval_games, logger)
    logger.info(f"Initial win rate vs random: {initial_eval['win_rate']:.1%}")
    logger.info(f"Model wins: {initial_eval['model_wins']}, "
               f"Random wins: {initial_eval['random_wins']}, "
               f"Draws: {initial_eval['draws']}")
    
    eval_history.append({
        'iteration': 0,
        **initial_eval
    })
    
    # Main training loop
    show_progress = not args.no_progress
    
    for iteration in range(start_iteration, args.iterations):
        logger.info("=" * 70)
        logger.info(f"ITERATION {iteration + 1}/{args.iterations}")
        logger.info("=" * 70)
        
        # Phase 1: Self-play data generation
        logger.info("-" * 40)
        logger.info("Phase 1: Self-play data generation")
        logger.info("-" * 40)
        
        self_play_data = run_self_play(
            model, config, config.num_self_play_games, logger
        )
        
        # Add to cumulative replay buffer
        replay_buffer.add(self_play_data)
        logger.info(f"New examples: {len(self_play_data)}, "
                   f"Total in buffer: {len(replay_buffer)}")
        
        # Phase 2: Training on accumulated data
        logger.info("-" * 40)
        logger.info("Phase 2: Training on accumulated data")
        logger.info("-" * 40)
        
        training_data = replay_buffer.get_all()
        iteration_history = trainer.train_iteration(
            training_data, num_epochs=config.num_epochs
        )
        
        logger.info(f"Training complete. Final loss: {iteration_history['train_loss'][-1]:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            config.checkpoint_dir,
            f'checkpoint_iter{iteration + 1}_{create_timestamp()}.pt'
        )
        save_checkpoint(
            checkpoint_path,
            model,
            config=config.to_dict(),
            iteration=iteration,
            global_epoch=trainer.global_epoch,
            replay_buffer_size=len(replay_buffer)
        )
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Phase 3: Evaluation (at specified intervals)
        if (iteration + 1) % args.eval_interval == 0:
            logger.info("-" * 40)
            logger.info("Phase 3: Evaluation vs Random Player")
            logger.info("-" * 40)
            
            eval_results = evaluate_against_random(
                model, eval_config, args.eval_games, logger
            )
            
            logger.info(f"Win rate vs random: {eval_results['win_rate']:.1%}")
            logger.info(f"Model wins: {eval_results['model_wins']}, "
                       f"Random wins: {eval_results['random_wins']}, "
                       f"Draws: {eval_results['draws']}")
            
            eval_history.append({
                'iteration': iteration + 1,
                **eval_results
            })
            
            # Check if model is improving
            if len(eval_history) >= 2:
                prev_win_rate = eval_history[-2]['win_rate']
                curr_win_rate = eval_history[-1]['win_rate']
                improvement = curr_win_rate - prev_win_rate
                if improvement > 0:
                    logger.info(f"✓ Improvement: +{improvement:.1%} from previous iteration")
                elif improvement < 0:
                    logger.info(f"↓ Regression: {improvement:.1%} from previous iteration")
                else:
                    logger.info("= No change from previous iteration")
    
    # Final evaluation with more games
    logger.info("=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)
    
    final_eval = evaluate_against_random(model, eval_config, args.eval_games * 2, logger)
    logger.info(f"Final win rate vs random: {final_eval['win_rate']:.1%}")
    logger.info(f"Model wins: {final_eval['model_wins']}, "
               f"Random wins: {final_eval['random_wins']}, "
               f"Draws: {final_eval['draws']}")
    
    # Save final model
    final_model_path = os.path.join(
        config.checkpoint_dir,
        f'final_model_{create_timestamp()}.pt'
    )
    save_checkpoint(
        final_model_path,
        model,
        config=config.to_dict(),
        iteration=args.iterations - 1,
        final_win_rate=final_eval['win_rate'],
        global_epoch=trainer.global_epoch
    )
    logger.info(f"Saved final model: {final_model_path}")
    
    # Generate plots
    training_summary = trainer.get_training_summary()
    plot_path = plot_progressive_results(eval_history, training_summary, config.log_dir)
    if plot_path:
        logger.info(f"Saved results plot: {plot_path}")
    
    # Save experiment results
    results = {
        'config': {
            'iterations': args.iterations,
            'games_per_iteration': config.num_self_play_games,
            'simulations': config.num_simulations,
            'epochs': config.num_epochs,
            'eval_games': args.eval_games,
            'res_blocks': config.num_res_blocks,
            'channels': config.num_channels,
            'learning_rate': config.learning_rate,
        },
        'eval_history': [
            {
                'iteration': h['iteration'],
                'win_rate': h['win_rate'],
                'model_wins': h['model_wins'],
                'random_wins': h['random_wins'],
                'draws': h['draws'],
                'avg_moves': h['avg_moves'],
            }
            for h in eval_history
        ],
        'training_summary': {
            'global_epoch': training_summary['global_epoch'],
            'global_step': training_summary['global_step'],
        },
        'final_model_path': final_model_path,
        'final_win_rate': final_eval['win_rate']
    }
    
    results_path = os.path.join(args.output_dir, f'experiment_results_{create_timestamp()}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved experiment results: {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\nTraining iterations: {args.iterations}")
    print(f"Total epochs trained: {trainer.global_epoch}")
    print(f"Total examples in replay buffer: {len(replay_buffer)}")
    print(f"\nFinal win rate vs random: {final_eval['win_rate']:.1%}")
    print(f"Final model saved to: {final_model_path}")
    
    # Check for overall improvement
    initial_win_rate = eval_history[0]['win_rate']
    final_win_rate = eval_history[-1]['win_rate']
    overall_improvement = final_win_rate - initial_win_rate
    
    print(f"\nProgressive Improvement:")
    print(f"  Initial win rate: {initial_win_rate:.1%}")
    print(f"  Final win rate:   {final_win_rate:.1%}")
    print(f"  Improvement:      {overall_improvement:+.1%}")
    
    if overall_improvement > 0:
        print("\n✓ SUCCESS: Model showed progressive improvement during training!")
    elif overall_improvement == 0:
        print("\n→ Model performance remained stable during training.")
    else:
        print("\n✗ Model did not show improvement. Consider adjusting hyperparameters.")
    
    # Print evaluation history table
    headers = ['Iteration', 'Win Rate', 'Model Wins', 'Random Wins', 'Draws']
    table = [
        [h['iteration'], f"{h['win_rate']:.1%}", h['model_wins'], h['random_wins'], h['draws']]
        for h in eval_history
    ]
    print("\nEvaluation History:")
    print(tabulate(table, headers=headers, tablefmt='simple'))
    
    logger.info("Experiment complete!")


if __name__ == '__main__':
    main()
