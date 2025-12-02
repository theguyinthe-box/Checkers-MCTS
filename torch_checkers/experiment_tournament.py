#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# torch_checkers/experiment_tournament.py
#
# Experiment 1: Multi-Player Tournament Training
#
# This script trains multiple Checkers players for a specified number of
# iterations each, saves their parameters, and runs a tournament-style
# match-up to select the best model.
#
# Features:
# - Train N different players independently
# - Save checkpoints with clear labels
# - Run tournament between all trained models
# - Select best model based on tournament results
#
# Usage:
#   python -m torch_checkers.experiment_tournament \
#       --num-players 3 \
#       --iterations 5 \
#       --games-per-iteration 50 \
#       --tournament-games 10 \
#       --first-to 3
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
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_checkers.config import Config, get_debug_config
from torch_checkers.model import CheckersModel, create_model
from torch_checkers.mcts import MCTSPlayer
from torch_checkers.dataset import ReplayBuffer
from torch_checkers.trainer import Trainer
from torch_checkers.train import run_self_play
from torch_checkers.utils import (
    setup_logging,
    set_seed,
    save_checkpoint,
    create_timestamp,
    get_gpu_memory_info
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
        description='Train multiple Checkers players and run tournament'
    )
    
    # Number of players
    parser.add_argument(
        '--num-players', type=int, default=3,
        help='Number of different players to train (default: 3)'
    )
    
    # Training parameters
    parser.add_argument(
        '--iterations', type=int, default=5,
        help='Number of training iterations per player (default: 5)'
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
        '--epochs', type=int, default=10,
        help='Training epochs per iteration (default: 10)'
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
        help='Learning rate (default: 1e-3)'
    )
    
    # Tournament parameters
    parser.add_argument(
        '--tournament-games', type=int, default=10,
        help='Number of games per match in tournament (default: 10)'
    )
    parser.add_argument(
        '--tournament-simulations', type=int, default=200,
        help='MCTS simulations per move in tournament (default: 200)'
    )
    parser.add_argument(
        '--first-to', type=int, default=3,
        help='First to N wins takes the match (default: 3)'
    )
    
    # System parameters
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for training (default: cuda if available)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Base random seed (each player gets seed + player_id)'
    )
    parser.add_argument(
        '--workers', type=int, default=2,
        help='Number of data loading workers (default: 2)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir', type=str, default='data/tournament_experiment',
        help='Directory for outputs (default: data/tournament_experiment)'
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


def create_player_config(args, player_id: int) -> Config:
    """Create configuration for a specific player."""
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
            seed=args.seed + player_id,
            
            # Directories - player specific
            checkpoint_dir=os.path.join(args.output_dir, f'player_{player_id}', 'checkpoints'),
            data_dir=os.path.join(args.output_dir, f'player_{player_id}', 'data'),
            log_dir=os.path.join(args.output_dir, f'player_{player_id}', 'logs'),
        )
    
    config.show_progress = not args.no_progress
    return config


def train_single_player(
    player_id: int,
    args,
    logger
) -> str:
    """
    Train a single player for the specified number of iterations.
    
    Returns the path to the final model checkpoint.
    """
    logger.info(f"=" * 60)
    logger.info(f"Training Player {player_id}")
    logger.info(f"=" * 60)
    
    config = create_player_config(args, player_id)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set seed
    set_seed(config.seed)
    
    # Create model
    model = create_model(config)
    logger.info(f"Player {player_id}: Model parameters: {model.count_parameters():,}")
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)
    
    # Initialize trainer once (preserves optimizer state across iterations)
    trainer = Trainer(model, config)
    
    # Training loop
    iterations_pbar = tqdm(
        range(args.iterations),
        desc=f"Player {player_id} iterations",
        disable=args.no_progress
    )
    
    for iteration in iterations_pbar:
        logger.info(f"Player {player_id} - Iteration {iteration + 1}/{args.iterations}")
        
        # Self-play phase
        self_play_data = run_self_play(
            model, config, config.num_self_play_games, logger
        )
        
        replay_buffer.add(self_play_data)
        
        # Training phase
        training_data = replay_buffer.get_all()
        trainer.train(training_data, num_epochs=config.num_epochs, val_split=0.2)
        
        iterations_pbar.set_postfix({
            'buffer': len(replay_buffer)
        })
    
    # Save final model
    final_model_path = os.path.join(
        config.checkpoint_dir,
        f'player_{player_id}_final_{create_timestamp()}.pt'
    )
    save_checkpoint(
        final_model_path,
        model,
        config=config.to_dict(),
        player_id=player_id,
        iterations=args.iterations
    )
    logger.info(f"Player {player_id}: Saved final model to {final_model_path}")
    
    return final_model_path


def play_match(
    model1: CheckersModel,
    model2: CheckersModel,
    config: Config,
    num_games: int,
    first_to: int,
    logger
) -> Dict:
    """
    Play a match between two models (first to N wins).
    
    Returns match results dictionary.
    """
    from tqdm import tqdm
    
    results = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0, 'games': []}
    
    games_pbar = tqdm(
        range(num_games),
        desc="Match games",
        disable=not config.show_progress,
        leave=False
    )
    
    for game_num in games_pbar:
        # Check if match is already decided
        if results['model1_wins'] >= first_to or results['model2_wins'] >= first_to:
            break
        
        game_env = Checkers()
        
        # Alternate who plays first
        if game_num % 2 == 0:
            p1_model, p2_model = model1, model2
            p1_label, p2_label = 'model1', 'model2'
        else:
            p1_model, p2_model = model2, model1
            p1_label, p2_label = 'model2', 'model1'
        
        mcts_p1 = MCTSPlayer(game_env, p1_model, config, config.device)
        mcts_p2 = MCTSPlayer(game_env, p2_model, config, config.device)
        
        move_count = 0
        while not game_env.done and move_count < config.max_game_moves:
            current_player = int(game_env.state[4, 0, 0])
            mcts = mcts_p1 if current_player == 0 else mcts_p2
            
            next_state, _, _ = mcts.get_action_probs(
                game_env.state,
                game_env.history,
                temperature=0,
                add_noise=False
            )
            
            game_env.step(next_state)
            move_count += 1
        
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
            if p1_label == 'model1':
                results['model1_wins'] += 1
            else:
                results['model2_wins'] += 1
        else:
            if p2_label == 'model1':
                results['model1_wins'] += 1
            else:
                results['model2_wins'] += 1
        
        results['games'].append({
            'game_num': game_num + 1,
            'p1': p1_label,
            'outcome': outcome,
            'moves': move_count
        })
        
        games_pbar.set_postfix({
            'M1': results['model1_wins'],
            'M2': results['model2_wins'],
            'D': results['draws']
        })
    
    return results


def run_tournament(
    model_paths: List[str],
    args,
    logger
) -> Tuple[int, Dict]:
    """
    Run round-robin tournament between all models.
    
    Returns (winner_index, tournament_results).
    """
    from tqdm import tqdm
    
    logger.info("=" * 60)
    logger.info("TOURNAMENT PHASE")
    logger.info("=" * 60)
    
    # Load all models
    models = []
    for i, path in enumerate(model_paths):
        checkpoint = torch.load(path, map_location=args.device)
        if 'config' in checkpoint:
            config = Config.from_dict(checkpoint['config'])
        else:
            config = create_player_config(args, i)
        config.device = args.device
        config.num_simulations = args.tournament_simulations
        config.show_progress = not args.no_progress
        
        model = CheckersModel(config).to(config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append((model, config))
        logger.info(f"Loaded Player {i}: {path}")
    
    # Tournament standings
    standings = {i: {'points': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'matches_won': 0}
                 for i in range(len(models))}
    match_results = []
    
    # Round-robin tournament
    num_matches = len(models) * (len(models) - 1) // 2
    match_num = 0
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            match_num += 1
            logger.info(f"\nMatch {match_num}/{num_matches}: Player {i} vs Player {j}")
            
            result = play_match(
                models[i][0], models[j][0],
                models[i][1],  # Use first model's config
                args.tournament_games,
                args.first_to,
                logger
            )
            
            # Update standings
            standings[i]['wins'] += result['model1_wins']
            standings[i]['losses'] += result['model2_wins']
            standings[i]['draws'] += result['draws']
            standings[i]['points'] += result['model1_wins'] + 0.5 * result['draws']
            
            standings[j]['wins'] += result['model2_wins']
            standings[j]['losses'] += result['model1_wins']
            standings[j]['draws'] += result['draws']
            standings[j]['points'] += result['model2_wins'] + 0.5 * result['draws']
            
            # Determine match winner
            if result['model1_wins'] >= args.first_to:
                standings[i]['matches_won'] += 1
                match_winner = f"Player {i}"
            elif result['model2_wins'] >= args.first_to:
                standings[j]['matches_won'] += 1
                match_winner = f"Player {j}"
            else:
                match_winner = "Tie"
            
            match_results.append({
                'player1': i,
                'player2': j,
                'result': result,
                'winner': match_winner
            })
            
            logger.info(f"Result: Player {i}: {result['model1_wins']} - "
                       f"Player {j}: {result['model2_wins']} "
                       f"(Draws: {result['draws']}) - Winner: {match_winner}")
    
    # Determine tournament winner
    sorted_standings = sorted(
        standings.items(),
        key=lambda x: (x[1]['matches_won'], x[1]['points']),
        reverse=True
    )
    winner_id = sorted_standings[0][0]
    
    return winner_id, {
        'standings': standings,
        'matches': match_results,
        'winner': winner_id
    }


def print_tournament_results(results: Dict, model_paths: List[str]):
    """Print formatted tournament results."""
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    
    # Standings table
    headers = ['Rank', 'Player', 'Matches Won', 'Points', 'W', 'L', 'D']
    table = []
    
    sorted_standings = sorted(
        results['standings'].items(),
        key=lambda x: (x[1]['matches_won'], x[1]['points']),
        reverse=True
    )
    
    for rank, (player_id, stats) in enumerate(sorted_standings, 1):
        table.append([
            rank,
            f"Player {player_id}",
            stats['matches_won'],
            stats['points'],
            stats['wins'],
            stats['losses'],
            stats['draws']
        ])
    
    print("\nFinal Standings:")
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))
    
    print(f"\n🏆 WINNER: Player {results['winner']}")
    print(f"   Model path: {model_paths[results['winner']]}")


def main():
    """Main experiment function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Multi-Player Tournament Training")
    logger.info("=" * 60)
    logger.info(f"Number of players: {args.num_players}")
    logger.info(f"Training iterations per player: {args.iterations}")
    logger.info(f"Self-play games per iteration: {args.games_per_iteration}")
    logger.info(f"Tournament games per match: {args.tournament_games}")
    logger.info(f"First to {args.first_to} wins takes the match")
    logger.info(f"Device: {args.device}")
    
    if args.device == 'cuda':
        mem_info = get_gpu_memory_info()
        logger.info(f"GPU Memory: {mem_info['total']:.1f} GB total")
    
    # Phase 1: Train all players
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: TRAINING PLAYERS")
    logger.info("=" * 60)
    
    model_paths = []
    for player_id in range(args.num_players):
        model_path = train_single_player(player_id, args, logger)
        model_paths.append(model_path)
    
    # Phase 2: Tournament
    winner_id, tournament_results = run_tournament(model_paths, args, logger)
    
    # Print and save results
    print_tournament_results(tournament_results, model_paths)
    
    # Save tournament results
    results_file = os.path.join(args.output_dir, f'tournament_results_{create_timestamp()}.json')
    import json
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON
        results_json = {
            'winner': winner_id,
            'winner_model_path': model_paths[winner_id],
            'model_paths': model_paths,
            'standings': {str(k): v for k, v in tournament_results['standings'].items()},
            'config': {
                'num_players': args.num_players,
                'iterations': args.iterations,
                'games_per_iteration': args.games_per_iteration,
                'tournament_games': args.tournament_games,
                'first_to': args.first_to
            }
        }
        json.dump(results_json, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info("Experiment complete!")


if __name__ == '__main__':
    main()
