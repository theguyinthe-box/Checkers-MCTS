#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# torch_checkers/experiment_single.py
#
# Experiment 2: Single Player Training with Evaluation
#
# This script trains a single Checkers player for a specified number of
# iterations with comprehensive evaluation and performance tracking.
#
# Features:
# - Train one player for N iterations
# - Evaluate against random player at regular intervals
# - Track and plot performance over time
# - Save checkpoints at each iteration
#
# Usage:
#   python -m torch_checkers.experiment_single \
#       --iterations 10 \
#       --games-per-iteration 100 \
#       --eval-games 20 \
#       --eval-interval 2
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
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_checkers.config import Config, get_debug_config
from torch_checkers.model import CheckersModel, create_model
from torch_checkers.mcts import MCTSPlayer
from torch_checkers.dataset import ReplayBuffer
from torch_checkers.trainer import Trainer
from torch_checkers.random_player import RandomPlayer
from torch_checkers.train import run_self_play
from torch_checkers.utils import (
    setup_logging,
    set_seed,
    save_checkpoint,
    create_timestamp,
    get_gpu_memory_info,
    plot_training_history
)

from Checkers import Checkers

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers=None, tablefmt=None, showindex=None):
        lines = []
        if headers:
            lines.append('\t'.join(str(h) for h in headers))
        for row in data:
            lines.append('\t'.join(str(c) for c in row))
        return '\n'.join(lines)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a single Checkers player with evaluation'
    )
    
    # Training parameters
    parser.add_argument(
        '--iterations', type=int, default=10,
        help='Number of training iterations (default: 10)'
    )
    parser.add_argument(
        '--games-per-iteration', type=int, default=100,
        help='Number of self-play games per iteration (default: 100)'
    )
    parser.add_argument(
        '--simulations', type=int, default=200,
        help='MCTS simulations per move during training (default: 200)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Training epochs per iteration (default: 10)'
    )
    parser.add_argument(
        '--max-moves', type=int, default=200,
        help='Maximum moves per game (default: 200)'
    )
    
    # Model parameters
    parser.add_argument(
        '--res-blocks', type=int, default=10,
        help='Number of residual blocks (default: 10)'
    )
    parser.add_argument(
        '--channels', type=int, default=128,
        help='Number of channels in convolutions (default: 128)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Training batch size (default: 256)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate (default: 1e-3)'
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
        '--eval-simulations', type=int, default=200,
        help='MCTS simulations per move during evaluation (default: 200)'
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
        '--workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/single_player_experiment',
        help='Directory for outputs (default: data/single_player_experiment)'
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
    else:
        config = Config(
            # Model
            num_res_blocks=args.res_blocks,
            num_channels=args.channels,
            
            # Training
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            
            # MCTS
            num_simulations=args.simulations,
            max_game_moves=args.max_moves,
            
            # Self-play
            num_self_play_games=args.games_per_iteration,
            
            # System
            device=args.device,
            num_workers=args.workers,
            seed=args.seed,
            
            # Directories
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
            data_dir=os.path.join(args.output_dir, 'data'),
            log_dir=os.path.join(args.output_dir, 'logs'),
        )
    
    config.show_progress = not args.no_progress
    return config


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
    from tqdm import tqdm
    
    model.eval()
    
    results = {
        'model_wins': 0,
        'random_wins': 0,
        'draws': 0,
        'games': [],
        'total_moves': 0
    }
    
    games_pbar = tqdm(
        range(num_games),
        desc="Evaluation vs Random",
        disable=not config.show_progress,
        unit="game"
    )
    
    for game_num in games_pbar:
        game_env = Checkers()
        
        # Alternate who plays first
        if game_num % 2 == 0:
            model_is_p1 = True
            mcts_player = MCTSPlayer(game_env, model, config, config.device)
            random_player = RandomPlayer(game_env)
        else:
            model_is_p1 = False
            random_player = RandomPlayer(game_env)
            mcts_player = MCTSPlayer(game_env, model, config, config.device)
        
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


def plot_evaluation_history(
    eval_history: List[Dict],
    save_path: str
):
    """Plot evaluation performance over training iterations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    iterations = [h['iteration'] for h in eval_history]
    win_rates = [h['win_rate'] for h in eval_history]
    model_wins = [h['model_wins'] for h in eval_history]
    random_wins = [h['random_wins'] for h in eval_history]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Win rate over time
    axes[0].plot(iterations, win_rates, 'b-o', linewidth=2, markersize=8)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
    axes[0].set_xlabel('Training Iteration')
    axes[0].set_ylabel('Win Rate vs Random')
    axes[0].set_title('Model Performance Over Training')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True)
    
    # Wins breakdown
    x = np.arange(len(iterations))
    width = 0.35
    axes[1].bar(x - width/2, model_wins, width, label='Model Wins', color='green')
    axes[1].bar(x + width/2, random_wins, width, label='Random Wins', color='red')
    axes[1].set_xlabel('Training Iteration')
    axes[1].set_ylabel('Number of Wins')
    axes[1].set_title('Win Distribution Over Training')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(iterations)
    axes[1].legend()
    axes[1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """Main experiment function."""
    args = parse_args()
    config = create_config(args)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config.log_dir)
    
    # Set seed
    if config.seed is not None:
        set_seed(config.seed)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Single Player Training with Evaluation")
    logger.info("=" * 60)
    logger.info(f"Training iterations: {args.iterations}")
    logger.info(f"Self-play games per iteration: {args.games_per_iteration}")
    logger.info(f"MCTS simulations per move: {args.simulations}")
    logger.info(f"Evaluation games: {args.eval_games}")
    logger.info(f"Evaluation interval: every {args.eval_interval} iterations")
    logger.info(f"Device: {config.device}")
    
    if config.device == 'cuda':
        mem_info = get_gpu_memory_info()
        logger.info(f"GPU Memory: {mem_info['total']:.1f} GB total")
    
    # Create model
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
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)
    
    # Evaluation history
    eval_history = []
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'iterations': []
    }
    
    # Initial evaluation (iteration 0)
    logger.info("\n" + "-" * 40)
    logger.info("Initial Evaluation (before training)")
    logger.info("-" * 40)
    
    eval_config = Config.from_dict(config.to_dict())
    eval_config.num_simulations = args.eval_simulations
    
    initial_eval = evaluate_against_random(model, eval_config, args.eval_games, logger)
    logger.info(f"Initial win rate vs random: {initial_eval['win_rate']:.1%}")
    logger.info(f"Model wins: {initial_eval['model_wins']}, Random wins: {initial_eval['random_wins']}, "
               f"Draws: {initial_eval['draws']}")
    
    eval_history.append({
        'iteration': 0,
        **initial_eval
    })
    
    # Training loop
    iterations_pbar = tqdm(
        range(start_iteration, args.iterations),
        desc="Training iterations",
        disable=args.no_progress
    )
    
    for iteration in iterations_pbar:
        logger.info("\n" + "=" * 60)
        logger.info(f"ITERATION {iteration + 1}/{args.iterations}")
        logger.info("=" * 60)
        
        # Phase 1: Self-play
        logger.info("-" * 40)
        logger.info("Phase 1: Self-play")
        logger.info("-" * 40)
        
        self_play_data = run_self_play(
            model, config, config.num_self_play_games, logger
        )
        
        replay_buffer.add(self_play_data)
        logger.info(f"Replay buffer size: {len(replay_buffer)}")
        
        # Save self-play data
        data_path = os.path.join(
            config.data_dir,
            f'selfplay_data_iter{iteration + 1}_{create_timestamp()}.pkl'
        )
        replay_buffer.save(data_path)
        
        # Phase 2: Training
        logger.info("-" * 40)
        logger.info("Phase 2: Training")
        logger.info("-" * 40)
        
        training_data = replay_buffer.get_all()
        trainer = Trainer(model, config)
        history = trainer.train(training_data, num_epochs=config.num_epochs, val_split=0.2)
        
        # Record training history
        if history['train_loss']:
            training_history['train_loss'].append(history['train_loss'][-1])
        if history['val_loss']:
            training_history['val_loss'].append(history['val_loss'][-1])
        training_history['iterations'].append(iteration + 1)
        
        # Save iteration checkpoint
        checkpoint_path = os.path.join(
            config.checkpoint_dir,
            f'checkpoint_iter{iteration + 1}_{create_timestamp()}.pt'
        )
        save_checkpoint(
            checkpoint_path,
            model,
            config=config.to_dict(),
            iteration=iteration,
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
        
        iterations_pbar.set_postfix({
            'buffer': len(replay_buffer),
            'win_rate': f"{eval_history[-1]['win_rate']:.1%}" if eval_history else "N/A"
        })
    
    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    
    final_eval = evaluate_against_random(model, eval_config, args.eval_games * 2, logger)
    logger.info(f"Final win rate vs random: {final_eval['win_rate']:.1%}")
    logger.info(f"Model wins: {final_eval['model_wins']}, Random wins: {final_eval['random_wins']}, "
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
        final_win_rate=final_eval['win_rate']
    )
    logger.info(f"Saved final model: {final_model_path}")
    
    # Plot evaluation history
    eval_plot_path = os.path.join(config.log_dir, f'evaluation_history_{create_timestamp()}.png')
    plot_evaluation_history(eval_history, eval_plot_path)
    logger.info(f"Saved evaluation plot: {eval_plot_path}")
    
    # Save experiment results
    results = {
        'config': {
            'iterations': args.iterations,
            'games_per_iteration': args.games_per_iteration,
            'simulations': args.simulations,
            'epochs': args.epochs,
            'eval_games': args.eval_games,
            'res_blocks': args.res_blocks,
            'channels': args.channels
        },
        'eval_history': eval_history,
        'training_history': training_history,
        'final_model_path': final_model_path,
        'final_win_rate': final_eval['win_rate']
    }
    
    results_path = os.path.join(args.output_dir, f'experiment_results_{create_timestamp()}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved experiment results: {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\nTraining iterations: {args.iterations}")
    print(f"Final win rate vs random: {final_eval['win_rate']:.1%}")
    print(f"Final model saved to: {final_model_path}")
    
    # Print evaluation history table
    headers = ['Iteration', 'Win Rate', 'Model Wins', 'Random Wins', 'Draws']
    table = [
        [h['iteration'], f"{h['win_rate']:.1%}", h['model_wins'], h['random_wins'], h['draws']]
        for h in eval_history
    ]
    print("\nEvaluation History:")
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))
    
    logger.info("Experiment complete!")


if __name__ == '__main__':
    main()
