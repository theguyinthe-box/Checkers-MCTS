#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# torch/evaluate.py
#
# Evaluation and tournament script for Checkers MCTS models.
#
# This script provides tools for:
# - Evaluating trained models against each other
# - Running tournaments between multiple models
# - Benchmarking model performance
# - Statistical analysis of results
#
# Usage:
#   python -m torch_checkers.evaluate --model1 path/to/model1.pt --model2 path/to/model2.pt
#   python -m torch_checkers.evaluate --tournament model1.pt model2.pt model3.pt
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
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_checkers.config import Config
from torch_checkers.model import CheckersModel
from torch_checkers.mcts import MCTSPlayer
from torch_checkers.utils import setup_logging, create_timestamp

# Import Checkers game environment
from Checkers import Checkers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Checkers MCTS models'
    )
    
    # Model paths
    parser.add_argument(
        '--model1', type=str, default=None,
        help='Path to first model checkpoint'
    )
    parser.add_argument(
        '--model2', type=str, default=None,
        help='Path to second model checkpoint'
    )
    
    # Tournament mode
    parser.add_argument(
        '--tournament', type=str, nargs='+', default=None,
        help='Paths to models for tournament'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--games', type=int, default=20,
        help='Number of games per match'
    )
    parser.add_argument(
        '--simulations', type=int, default=200,
        help='MCTS simulations per move'
    )
    parser.add_argument(
        '--max-moves', type=int, default=200,
        help='Maximum moves per game'
    )
    
    # Output
    parser.add_argument(
        '--output-dir', type=str, default='data/evaluation_results',
        help='Directory for evaluation results'
    )
    
    # System
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    device: str = 'cuda'
) -> Tuple[CheckersModel, Config]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model to.
        
    Returns:
        Tuple of (model, config).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        config = Config.from_dict(checkpoint['config'])
    else:
        config = Config()
    
    config.device = device
    model = CheckersModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def play_game(
    model1: CheckersModel,
    model2: CheckersModel,
    config: Config,
    logger=None
) -> Tuple[str, int]:
    """
    Play a single game between two models.
    
    Args:
        model1: Model playing as player 1.
        model2: Model playing as player 2.
        config: Configuration object.
        logger: Optional logger.
        
    Returns:
        Tuple of (outcome, move_count).
    """
    game_env = Checkers()
    
    mcts1 = MCTSPlayer(game_env, model1, config, config.device)
    mcts2 = MCTSPlayer(game_env, model2, config, config.device)
    
    move_count = 0
    
    while not game_env.done and move_count < config.max_game_moves:
        current_player = int(game_env.state[4, 0, 0])
        mcts = mcts1 if current_player == 0 else mcts2
        
        next_state, _, _ = mcts.get_action_probs(
            game_env.state,
            game_env.history,
            temperature=0,  # Greedy play
            add_noise=False
        )
        
        game_env.step(next_state)
        move_count += 1
    
    # Determine outcome
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
    
    return outcome, move_count


def evaluate_match(
    model1: CheckersModel,
    model2: CheckersModel,
    config: Config,
    num_games: int,
    model1_name: str = 'Model1',
    model2_name: str = 'Model2',
    logger=None
) -> Dict:
    """
    Run a match between two models.
    
    Each model plays half the games as player 1 and half as player 2.
    
    Args:
        model1: First model.
        model2: Second model.
        config: Configuration.
        num_games: Total number of games.
        model1_name: Name for model1 in output.
        model2_name: Name for model2 in output.
        logger: Optional logger.
        
    Returns:
        Dictionary with match statistics.
    """
    results = {
        model1_name: {'wins': 0, 'losses': 0, 'draws': 0},
        model2_name: {'wins': 0, 'losses': 0, 'draws': 0},
        'games': [],
        'total_moves': 0
    }
    
    if logger:
        logger.info(f"Match: {model1_name} vs {model2_name} ({num_games} games)")
    
    for game_idx in range(num_games):
        # Alternate who plays first
        if game_idx % 2 == 0:
            p1_model, p2_model = model1, model2
            p1_name, p2_name = model1_name, model2_name
        else:
            p1_model, p2_model = model2, model1
            p1_name, p2_name = model2_name, model1_name
        
        outcome, move_count = play_game(p1_model, p2_model, config, logger)
        results['total_moves'] += move_count
        
        # Record result
        game_result = {
            'game_num': game_idx + 1,
            'p1': p1_name,
            'p2': p2_name,
            'outcome': outcome,
            'moves': move_count
        }
        results['games'].append(game_result)
        
        # Update statistics
        if outcome == 'draw':
            results[p1_name]['draws'] += 1
            results[p2_name]['draws'] += 1
        elif outcome == 'player1_wins':
            results[p1_name]['wins'] += 1
            results[p2_name]['losses'] += 1
        else:  # player2_wins
            results[p2_name]['wins'] += 1
            results[p1_name]['losses'] += 1
        
        if logger and (game_idx + 1) % 5 == 0:
            logger.info(f"  Progress: {game_idx + 1}/{num_games} games completed")
    
    # Calculate statistics
    for name in [model1_name, model2_name]:
        stats = results[name]
        total = stats['wins'] + stats['losses'] + stats['draws']
        stats['win_rate'] = (stats['wins'] + 0.5 * stats['draws']) / total
        stats['total_games'] = total
    
    results['avg_moves'] = results['total_moves'] / num_games
    
    return results


def run_tournament(
    model_paths: List[str],
    config: Config,
    num_games: int,
    logger
) -> Dict:
    """
    Run a round-robin tournament between multiple models.
    
    Args:
        model_paths: List of model checkpoint paths.
        config: Configuration.
        num_games: Games per match.
        logger: Logger instance.
        
    Returns:
        Tournament results dictionary.
    """
    # Load all models
    models = {}
    for path in model_paths:
        name = os.path.basename(path).replace('.pt', '')
        model, _ = load_model(path, config.device)
        models[name] = model
        logger.info(f"Loaded model: {name}")
    
    model_names = list(models.keys())
    
    # Initialize results
    standings = {name: {'points': 0, 'wins': 0, 'losses': 0, 'draws': 0} 
                 for name in model_names}
    match_results = []
    
    # Play all matches
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            logger.info(f"\nMatch: {name1} vs {name2}")
            
            result = evaluate_match(
                models[name1], models[name2], config, num_games,
                name1, name2, logger
            )
            match_results.append(result)
            
            # Update standings
            for name in [name1, name2]:
                stats = result[name]
                standings[name]['wins'] += stats['wins']
                standings[name]['losses'] += stats['losses']
                standings[name]['draws'] += stats['draws']
                standings[name]['points'] += stats['wins'] + 0.5 * stats['draws']
    
    # Sort standings by points
    sorted_standings = sorted(
        standings.items(),
        key=lambda x: x[1]['points'],
        reverse=True
    )
    
    return {
        'standings': dict(sorted_standings),
        'matches': match_results
    }


def print_match_results(results: Dict, model1_name: str, model2_name: str):
    """Print formatted match results."""
    print("\n" + "=" * 60)
    print(f"MATCH RESULTS: {model1_name} vs {model2_name}")
    print("=" * 60)
    
    # Summary table
    headers = ['Model', 'Wins', 'Losses', 'Draws', 'Win Rate']
    table = []
    for name in [model1_name, model2_name]:
        stats = results[name]
        table.append([
            name,
            stats['wins'],
            stats['losses'],
            stats['draws'],
            f"{stats['win_rate']:.1%}"
        ])
    
    print("\nSummary:")
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))
    
    # Game details
    print(f"\nAverage game length: {results['avg_moves']:.1f} moves")
    
    print("\nGame Details:")
    game_headers = ['Game', 'P1', 'P2', 'Outcome', 'Moves']
    game_table = [
        [g['game_num'], g['p1'], g['p2'], g['outcome'], g['moves']]
        for g in results['games']
    ]
    print(tabulate(game_table, headers=game_headers, tablefmt='simple'))


def print_tournament_results(results: Dict):
    """Print formatted tournament results."""
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    
    # Standings table
    headers = ['Rank', 'Model', 'Points', 'W', 'L', 'D']
    table = []
    for rank, (name, stats) in enumerate(results['standings'].items(), 1):
        table.append([
            rank,
            name,
            stats['points'],
            stats['wins'],
            stats['losses'],
            stats['draws']
        ])
    
    print("\nFinal Standings:")
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))


def save_results(results: Dict, output_path: str):
    """Save results to file."""
    import json
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create config
    config = Config(
        num_simulations=args.simulations,
        max_game_moves=args.max_moves,
        device=args.device
    )
    
    logger.info("=" * 60)
    logger.info("Checkers MCTS Model Evaluation")
    logger.info("=" * 60)
    
    if args.tournament:
        # Tournament mode
        logger.info(f"Tournament mode: {len(args.tournament)} models")
        logger.info(f"Games per match: {args.games}")
        
        results = run_tournament(
            args.tournament, config, args.games, logger
        )
        
        print_tournament_results(results)
        
        # Save results
        output_path = os.path.join(
            args.output_dir,
            f'tournament_{create_timestamp()}.json'
        )
        save_results(results, output_path)
        logger.info(f"Results saved to: {output_path}")
        
    elif args.model1 and args.model2:
        # Match mode
        logger.info(f"Match mode: {args.model1} vs {args.model2}")
        logger.info(f"Number of games: {args.games}")
        
        # Load models
        model1, _ = load_model(args.model1, config.device)
        model2, _ = load_model(args.model2, config.device)
        
        model1_name = os.path.basename(args.model1).replace('.pt', '')
        model2_name = os.path.basename(args.model2).replace('.pt', '')
        
        results = evaluate_match(
            model1, model2, config, args.games,
            model1_name, model2_name, logger
        )
        
        print_match_results(results, model1_name, model2_name)
        
        # Save results
        output_path = os.path.join(
            args.output_dir,
            f'match_{model1_name}_vs_{model2_name}_{create_timestamp()}.json'
        )
        save_results(results, output_path)
        logger.info(f"Results saved to: {output_path}")
        
    else:
        logger.error("Please provide either --model1 and --model2 for a match, "
                    "or --tournament for a tournament")
        return
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
