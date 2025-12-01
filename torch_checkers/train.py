#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# torch/train.py
#
# Main training script for Checkers MCTS with PyTorch neural network.
#
# This script implements the full AlphaZero-style training pipeline:
# 1. Self-play: Generate training data by playing games with MCTS
# 2. Training: Train neural network on self-play data
# 3. Evaluation: Compare new model against previous best
#
# Features:
# - GPU-optimized for 16GB VRAM
# - Mixed precision training
# - Configurable hyperparameters via command line or config
# - Checkpoint saving and resumption
# - Comprehensive logging
#
# Usage:
#   python -m torch_checkers.train [--iterations 10] [--games 100] [--simulations 200]
#
###############################################################################
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_checkers.config import Config, get_debug_config
from torch_checkers.model import CheckersModel, create_model
from torch_checkers.mcts import MCTSPlayer, run_self_play_game, run_self_play_game_with_progress
from torch_checkers.dataset import ReplayBuffer, CheckersDataset, create_data_loaders
from torch_checkers.trainer import Trainer
from torch_checkers.utils import (
    setup_logging, 
    set_seed, 
    save_checkpoint,
    create_timestamp,
    save_training_history,
    plot_training_history,
    get_gpu_memory_info
)

# Import Checkers game environment from parent directory
from Checkers import Checkers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Checkers MCTS with PyTorch neural network'
    )
    
    # Training iterations
    parser.add_argument(
        '--iterations', type=int, default=10,
        help='Number of training iterations (self-play + train cycles)'
    )
    
    # Self-play parameters
    parser.add_argument(
        '--games', type=int, default=100,
        help='Number of self-play games per iteration'
    )
    parser.add_argument(
        '--simulations', type=int, default=200,
        help='Number of MCTS simulations per move'
    )
    parser.add_argument(
        '--max-moves', type=int, default=200,
        help='Maximum moves per game before termination'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Training epochs per iteration'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate'
    )
    
    # Model parameters
    parser.add_argument(
        '--res-blocks', type=int, default=10,
        help='Number of residual blocks'
    )
    parser.add_argument(
        '--channels', type=int, default=128,
        help='Number of channels in convolutions'
    )
    
    # System parameters
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for training'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of data loading workers'
    )
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='data/torch_checkpoints',
        help='Directory to save checkpoints'
    )
    
    # Debug mode
    parser.add_argument(
        '--debug', action='store_true',
        help='Use debug configuration (small model, few games)'
    )
    
    # Parallelization options
    parser.add_argument(
        '--parallel-simulations', type=int, default=1,
        help='Number of parallel MCTS simulations (batched NN inference). '
             'Higher values utilize more GPU VRAM for faster training. '
             'Recommended: 4-16 depending on available VRAM.'
    )
    
    # Progress display options
    parser.add_argument(
        '--progress', action='store_true', default=True,
        help='Show progress bars during self-play and training (default: enabled)'
    )
    parser.add_argument(
        '--no-progress', action='store_true',
        help='Disable progress bars for cleaner log output'
    )
    
    return parser.parse_args()


def create_config(args) -> Config:
    """Create configuration from command line arguments."""
    # Determine if progress should be shown
    show_progress = args.progress and not args.no_progress
    
    if args.debug:
        config = get_debug_config()
        config.show_progress = show_progress
        config.parallel_simulations = args.parallel_simulations
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
            num_self_play_games=args.games,
            
            # Parallelization
            parallel_simulations=args.parallel_simulations,
            
            # Progress display
            show_progress=show_progress,
            
            # System
            device=args.device,
            num_workers=args.workers,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
        )
    
    return config


def run_self_play(
    model: CheckersModel,
    config: Config,
    num_games: int,
    logger
) -> List[Tuple]:
    """
    Generate training data through self-play.
    
    Args:
        model: Neural network model.
        config: Configuration object.
        num_games: Number of games to play.
        logger: Logger instance.
        
    Returns:
        List of training examples from all games.
    """
    from tqdm import tqdm
    
    model.eval()
    all_data = []
    
    parallel_info = f" (parallel sims: {config.parallel_simulations})" if config.parallel_simulations > 1 else ""
    logger.info(f"Starting self-play: {num_games} games with {config.num_simulations} simulations/move{parallel_info}")
    
    outcomes = {'player1_wins': 0, 'player2_wins': 0, 'draw': 0}
    total_moves = 0
    
    # Create progress bar for games
    show_progress = getattr(config, 'show_progress', True)
    games_pbar = tqdm(
        range(num_games), 
        desc="Self-play games",
        disable=not show_progress,
        unit="game"
    )
    
    for game_num in games_pbar:
        # Create fresh game environment
        game_env = Checkers()
        
        # Play game with progress tracking
        # Only show move-by-move progress for small game counts to avoid cluttered output
        detailed_threshold = getattr(config, 'detailed_game_progress_threshold', 10)
        game_data, outcome, move_count = run_self_play_game_with_progress(
            game_env, model, config, config.device,
            show_progress=show_progress and num_games <= detailed_threshold
        )
        
        all_data.extend(game_data)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total_moves += move_count
        
        # Update progress bar with current statistics
        games_pbar.set_postfix({
            'moves': move_count,
            'examples': len(all_data),
            'P1': outcomes['player1_wins'],
            'P2': outcomes['player2_wins'],
            'Draw': outcomes['draw']
        })
        
        if not show_progress and (game_num + 1) % 10 == 0:
            logger.info(
                f"Self-play progress: {game_num + 1}/{num_games} games, "
                f"{len(all_data)} training examples"
            )
    
    logger.info(
        f"Self-play complete. Outcomes: P1 wins: {outcomes['player1_wins']}, "
        f"P2 wins: {outcomes['player2_wins']}, Draws: {outcomes['draw']}. "
        f"Avg moves: {total_moves / num_games:.1f}"
    )
    
    return all_data


def train_model(
    model: CheckersModel,
    training_data: List[Tuple],
    config: Config,
    logger
) -> dict:
    """
    Train the model on self-play data.
    
    Args:
        model: Neural network model.
        training_data: List of training examples.
        config: Configuration object.
        logger: Logger instance.
        
    Returns:
        Training history dictionary.
    """
    logger.info(f"Starting training on {len(training_data)} examples")
    
    trainer = Trainer(model, config)
    history = trainer.train(
        training_data,
        num_epochs=config.num_epochs,
        val_split=config.validation_split
    )
    
    return history


def evaluate_models(
    new_model: CheckersModel,
    old_model: CheckersModel,
    config: Config,
    num_games: int,
    logger
) -> Tuple[float, dict]:
    """
    Evaluate new model against old model.
    
    Args:
        new_model: Newly trained model.
        old_model: Previous best model.
        config: Configuration object.
        num_games: Number of evaluation games.
        logger: Logger instance.
        
    Returns:
        Tuple of (win_rate, outcomes_dict).
    """
    from tqdm import tqdm
    
    logger.info(f"Evaluating models: {num_games} games")
    
    new_model.eval()
    old_model.eval()
    
    outcomes = {'new_wins': 0, 'old_wins': 0, 'draws': 0}
    
    # Create progress bar for evaluation games
    show_progress = getattr(config, 'show_progress', True)
    eval_pbar = tqdm(
        range(num_games),
        desc="Evaluation games",
        disable=not show_progress,
        unit="game"
    )
    
    for game_num in eval_pbar:
        game_env = Checkers()
        
        # Alternate which model plays first
        if game_num % 2 == 0:
            p1_model, p2_model = new_model, old_model
            p1_label, p2_label = 'new', 'old'
        else:
            p1_model, p2_model = old_model, new_model
            p1_label, p2_label = 'old', 'new'
        
        mcts_p1 = MCTSPlayer(game_env, p1_model, config, config.device)
        mcts_p2 = MCTSPlayer(game_env, p2_model, config, config.device)
        
        move_count = 0
        while not game_env.done and move_count < config.max_game_moves:
            current_player = int(game_env.state[4, 0, 0])
            mcts = mcts_p1 if current_player == 0 else mcts_p2
            
            next_state, _, _ = mcts.get_action_probs(
                game_env.state,
                game_env.history,
                temperature=0,  # Greedy during evaluation
                add_noise=False
            )
            
            game_env.step(next_state)
            move_count += 1
        
        # Determine winner
        if not game_env.done:
            # Terminated game - count pieces
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
            outcomes['draws'] += 1
        elif outcome == 'player1_wins':
            if p1_label == 'new':
                outcomes['new_wins'] += 1
            else:
                outcomes['old_wins'] += 1
        else:  # player2_wins
            if p2_label == 'new':
                outcomes['new_wins'] += 1
            else:
                outcomes['old_wins'] += 1
        
        # Update progress bar
        eval_pbar.set_postfix({
            'New': outcomes['new_wins'],
            'Old': outcomes['old_wins'],
            'Draw': outcomes['draws']
        })
    
    total_games = outcomes['new_wins'] + outcomes['old_wins'] + outcomes['draws']
    win_rate = (outcomes['new_wins'] + 0.5 * outcomes['draws']) / total_games
    
    logger.info(
        f"Evaluation complete. New model win rate: {win_rate:.2%}, "
        f"New wins: {outcomes['new_wins']}, Old wins: {outcomes['old_wins']}, "
        f"Draws: {outcomes['draws']}"
    )
    
    return win_rate, outcomes


def main():
    """Main training loop."""
    args = parse_args()
    config = create_config(args)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config.log_dir)
    
    # Set random seed
    if config.seed is not None:
        set_seed(config.seed)
        logger.info(f"Set random seed to {config.seed}")
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("Checkers MCTS Training with PyTorch")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Model: {config.num_res_blocks} res blocks, {config.num_channels} channels")
    logger.info(f"Training: {config.batch_size} batch size, {config.learning_rate} LR")
    logger.info(f"MCTS: {config.num_simulations} simulations")
    logger.info(f"Parallel simulations: {config.parallel_simulations}")
    logger.info(f"Self-play: {config.num_self_play_games} games per iteration")
    logger.info(f"Progress display: {'enabled' if config.show_progress else 'disabled'}")
    
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
    
    # Keep track of best model
    best_model_state = model.state_dict()
    
    # Main training loop
    for iteration in range(start_iteration, args.iterations):
        logger.info("=" * 60)
        logger.info(f"ITERATION {iteration + 1}/{args.iterations}")
        logger.info("=" * 60)
        
        # Phase 1: Self-play
        logger.info("-" * 40)
        logger.info("Phase 1: Self-play")
        logger.info("-" * 40)
        
        self_play_data = run_self_play(
            model, config, config.num_self_play_games, logger
        )
        
        # Add data to replay buffer
        replay_buffer.add(self_play_data)
        logger.info(f"Replay buffer size: {len(replay_buffer)}")
        
        # Save self-play data
        data_path = os.path.join(
            config.data_dir,
            f'selfplay_data_iter{iteration + 1}_{create_timestamp()}.pkl'
        )
        replay_buffer.save(data_path)
        logger.info(f"Saved self-play data: {data_path}")
        
        # Phase 2: Training
        logger.info("-" * 40)
        logger.info("Phase 2: Training")
        logger.info("-" * 40)
        
        training_data = replay_buffer.get_all()
        history = train_model(model, training_data, config, logger)
        
        # Save training history
        history_path = os.path.join(
            config.log_dir,
            f'history_iter{iteration + 1}_{create_timestamp()}.json'
        )
        save_training_history(history, history_path)
        
        # Plot training curves
        plot_path = os.path.join(
            config.log_dir,
            f'training_plot_iter{iteration + 1}_{create_timestamp()}.png'
        )
        plot_training_history(history, plot_path)
        
        # Phase 3: Evaluation
        logger.info("-" * 40)
        logger.info("Phase 3: Evaluation")
        logger.info("-" * 40)
        
        # Load best model for comparison
        old_model = CheckersModel(config).to(config.device)
        old_model.load_state_dict(best_model_state)
        
        win_rate, outcomes = evaluate_models(
            model, old_model, config, config.num_evaluation_games, logger
        )
        
        # Accept new model if it wins enough
        if win_rate >= config.win_threshold:
            logger.info(f"New model accepted (win rate: {win_rate:.2%})")
            best_model_state = model.state_dict()
            
            # Save best model
            best_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(
                best_path,
                model,
                epoch=0,
                best_val_loss=history['val_loss'][-1] if history['val_loss'] else 0,
                config=config.to_dict(),
                iteration=iteration,
                win_rate=win_rate
            )
            logger.info(f"Saved best model: {best_path}")
        else:
            logger.info(f"New model rejected (win rate: {win_rate:.2%})")
            # Revert to best model
            model.load_state_dict(best_model_state)
        
        # Save iteration checkpoint
        checkpoint_path = os.path.join(
            config.checkpoint_dir,
            f'checkpoint_iter{iteration + 1}_{create_timestamp()}.pt'
        )
        save_checkpoint(
            checkpoint_path,
            model,
            epoch=0,
            config=config.to_dict(),
            iteration=iteration,
            win_rate=win_rate,
            replay_buffer_size=len(replay_buffer)
        )
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
