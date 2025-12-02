#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# torch_checkers/evaluate_vs_random.py
#
# Evaluation Pipeline: Test trained models against Random Legal Move player
#
# This script provides a simple way to evaluate any trained model against
# the random player baseline. It supports:
# - Single model evaluation
# - Multiple models comparison
# - Configurable number of games and MCTS simulations
#
# Usage:
#   # Evaluate a single model
#   python -m torch_checkers.evaluate_vs_random --model path/to/model.pt --games 50
#
#   # Compare multiple models
#   python -m torch_checkers.evaluate_vs_random --models model1.pt model2.pt model3.pt
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_checkers.config import Config
from torch_checkers.model import CheckersModel
from torch_checkers.mcts import MCTSPlayer
from torch_checkers.random_player import RandomPlayer
from torch_checkers.utils import setup_logging, create_timestamp

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
        description='Evaluate trained Checkers models against random player'
    )
    
    # Model paths
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to a single model checkpoint to evaluate'
    )
    parser.add_argument(
        '--models', type=str, nargs='+', default=None,
        help='Paths to multiple model checkpoints to compare'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--games', type=int, default=50,
        help='Number of evaluation games per model (default: 50)'
    )
    parser.add_argument(
        '--simulations', type=int, default=200,
        help='MCTS simulations per move (default: 200)'
    )
    parser.add_argument(
        '--max-moves', type=int, default=200,
        help='Maximum moves per game (default: 200)'
    )
    
    # System parameters
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference (default: cuda if available)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir', type=str, default='data/evaluation_results',
        help='Directory for evaluation results (default: data/evaluation_results)'
    )
    
    # Progress
    parser.add_argument(
        '--no-progress', action='store_true',
        help='Disable progress bars'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> Tuple[CheckersModel, Config]:
    """Load model from checkpoint."""
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


def evaluate_model_vs_random(
    model: CheckersModel,
    config: Config,
    num_games: int,
    show_progress: bool = True
) -> Dict:
    """
    Evaluate a model against the random player.
    
    Args:
        model: Trained model to evaluate.
        config: Configuration for MCTS.
        num_games: Number of games to play.
        show_progress: Whether to show progress bar.
        
    Returns:
        Dictionary with evaluation results.
    """
    from tqdm import tqdm
    
    model.eval()
    
    results = {
        'model_wins': 0,
        'random_wins': 0,
        'draws': 0,
        'model_wins_as_p1': 0,
        'model_wins_as_p2': 0,
        'games': [],
        'total_moves': 0
    }
    
    games_pbar = tqdm(
        range(num_games),
        desc="Evaluation",
        disable=not show_progress,
        unit="game"
    )
    
    for game_num in games_pbar:
        game_env = Checkers()
        
        # Alternate who plays first
        model_is_p1 = (game_num % 2 == 0)
        
        if model_is_p1:
            mcts_player = MCTSPlayer(game_env, model, config, config.device)
            random_player = RandomPlayer(game_env)
        else:
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
        game_result = {
            'game_num': game_num + 1,
            'model_first': model_is_p1,
            'outcome': outcome,
            'moves': move_count
        }
        
        if outcome == 'draw':
            results['draws'] += 1
            game_result['winner'] = 'draw'
        elif outcome == 'player1_wins':
            if model_is_p1:
                results['model_wins'] += 1
                results['model_wins_as_p1'] += 1
                game_result['winner'] = 'model'
            else:
                results['random_wins'] += 1
                game_result['winner'] = 'random'
        else:  # player2_wins
            if not model_is_p1:
                results['model_wins'] += 1
                results['model_wins_as_p2'] += 1
                game_result['winner'] = 'model'
            else:
                results['random_wins'] += 1
                game_result['winner'] = 'random'
        
        results['games'].append(game_result)
        
        games_pbar.set_postfix({
            'Model': results['model_wins'],
            'Random': results['random_wins'],
            'Draw': results['draws']
        })
    
    # Calculate statistics
    total_games = results['model_wins'] + results['random_wins'] + results['draws']
    results['win_rate'] = (results['model_wins'] + 0.5 * results['draws']) / total_games
    results['pure_win_rate'] = results['model_wins'] / total_games
    results['avg_moves'] = results['total_moves'] / total_games
    results['total_games'] = total_games
    
    return results


def print_single_model_results(results: Dict, model_name: str):
    """Print results for a single model evaluation."""
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 60)
    
    print(f"\nOverall Performance:")
    print(f"  Win Rate (including draws as 0.5): {results['win_rate']:.1%}")
    print(f"  Pure Win Rate: {results['pure_win_rate']:.1%}")
    print(f"  Model Wins: {results['model_wins']}")
    print(f"  Random Wins: {results['random_wins']}")
    print(f"  Draws: {results['draws']}")
    print(f"  Total Games: {results['total_games']}")
    
    print(f"\nBy Position:")
    games_as_p1 = results['total_games'] // 2 + results['total_games'] % 2
    games_as_p2 = results['total_games'] // 2
    print(f"  As Player 1 (first): {results['model_wins_as_p1']} wins / {games_as_p1} games")
    print(f"  As Player 2 (second): {results['model_wins_as_p2']} wins / {games_as_p2} games")
    
    print(f"\nGame Statistics:")
    print(f"  Average game length: {results['avg_moves']:.1f} moves")


def print_comparison_results(all_results: Dict[str, Dict]):
    """Print comparison results for multiple models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    # Create comparison table
    headers = ['Model', 'Win Rate', 'Pure Win Rate', 'Model Wins', 'Random Wins', 'Draws', 'Avg Moves']
    table = []
    
    for model_name, results in all_results.items():
        table.append([
            model_name,
            f"{results['win_rate']:.1%}",
            f"{results['pure_win_rate']:.1%}",
            results['model_wins'],
            results['random_wins'],
            results['draws'],
            f"{results['avg_moves']:.1f}"
        ])
    
    # Sort by win rate
    table.sort(key=lambda x: float(x[1].rstrip('%')), reverse=True)
    
    print("\nRanking by Win Rate:")
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1]['win_rate'])
    print(f"\n🏆 Best Model: {best_model[0]} (Win Rate: {best_model[1]['win_rate']:.1%})")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Validate arguments
    if args.model is None and args.models is None:
        print("Error: Please provide --model or --models argument")
        return
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("Evaluation: Trained Models vs Random Player")
    logger.info("=" * 60)
    logger.info(f"Games per model: {args.games}")
    logger.info(f"MCTS simulations: {args.simulations}")
    logger.info(f"Device: {args.device}")
    
    # Determine which models to evaluate
    if args.model:
        model_paths = [args.model]
    else:
        model_paths = args.models
    
    logger.info(f"Models to evaluate: {len(model_paths)}")
    
    all_results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.pt', '')
        logger.info(f"\nEvaluating: {model_name}")
        
        # Load model
        try:
            model, config = load_model(model_path, args.device)
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            continue
        
        # Update config for evaluation
        config.num_simulations = args.simulations
        config.max_game_moves = args.max_moves
        config.show_progress = not args.no_progress
        
        # Evaluate
        results = evaluate_model_vs_random(
            model, config, args.games, show_progress=not args.no_progress
        )
        
        all_results[model_name] = results
        
        logger.info(f"  Win Rate: {results['win_rate']:.1%}")
        logger.info(f"  Model Wins: {results['model_wins']}, "
                   f"Random Wins: {results['random_wins']}, "
                   f"Draws: {results['draws']}")
    
    # Print results
    if len(all_results) == 1:
        model_name, results = list(all_results.items())[0]
        print_single_model_results(results, model_name)
    else:
        print_comparison_results(all_results)
    
    # Save results
    results_file = os.path.join(
        args.output_dir,
        f'evaluation_vs_random_{create_timestamp()}.json'
    )
    
    # Prepare results for JSON (remove non-serializable data)
    json_results = {}
    for model_name, results in all_results.items():
        json_results[model_name] = {
            'model_wins': results['model_wins'],
            'random_wins': results['random_wins'],
            'draws': results['draws'],
            'win_rate': results['win_rate'],
            'pure_win_rate': results['pure_win_rate'],
            'avg_moves': results['avg_moves'],
            'total_games': results['total_games'],
            'model_wins_as_p1': results['model_wins_as_p1'],
            'model_wins_as_p2': results['model_wins_as_p2']
        }
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'games': args.games,
                'simulations': args.simulations,
                'max_moves': args.max_moves
            },
            'results': json_results
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
