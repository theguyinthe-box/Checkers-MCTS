#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# torch/play.py
#
# Interactive play script for Checkers with trained PyTorch model.
#
# This script allows:
# - Human vs AI gameplay
# - AI vs AI demonstration games
# - Visualization of game state and AI thinking
#
# Usage:
#   python -m torch_checkers.play --model path/to/model.pt
#   python -m torch_checkers.play --model path/to/model.pt --human-first
#   python -m torch_checkers.play --model path/to/model.pt --ai-vs-ai
#
###############################################################################
"""

import argparse
import os
import sys
from typing import Optional, Tuple
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_checkers.config import Config
from torch_checkers.model import CheckersModel
from torch_checkers.mcts import MCTSPlayer

# Import Checkers game environment
from Checkers import Checkers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Play Checkers against trained model'
    )
    
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--human-first', action='store_true',
        help='Human plays as player 1 (first)'
    )
    parser.add_argument(
        '--ai-vs-ai', action='store_true',
        help='Watch AI play against itself'
    )
    parser.add_argument(
        '--simulations', type=int, default=400,
        help='MCTS simulations per move'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--delay', type=float, default=1.0,
        help='Delay between AI moves (seconds)'
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


def print_board(game_env: Checkers, highlight_moves: bool = True):
    """
    Print the current board state with optional move highlighting.
    
    Args:
        game_env: Checkers game environment.
        highlight_moves: Whether to show possible moves.
    """
    game_env.print_board()
    
    if highlight_moves and not game_env.done:
        legal_states = game_env.legal_next_states
        moves = states_to_moves(game_env.state, legal_states)
        
        print("\nAvailable moves:")
        for idx, (start, end) in enumerate(moves):
            print(f"  {idx + 1}. ({start[0]+1}, {start[1]+1}) -> ({end[0]+1}, {end[1]+1})")


def states_to_moves(current_state, next_states):
    """
    Convert list of next states to human-readable moves.
    
    Args:
        current_state: Current game state.
        next_states: List of possible next states.
        
    Returns:
        List of ((from_row, from_col), (to_row, to_col)) tuples.
    """
    moves = []
    board = current_state[0] + 2*current_state[1] + 3*current_state[2] + 4*current_state[3]
    
    for next_state in next_states:
        next_board = next_state[0] + 2*next_state[1] + 3*next_state[2] + 4*next_state[3]
        diff = board - next_board
        
        # Find where piece moved from (positive diff)
        from_positions = np.where(diff > 0)
        # Find where piece moved to (negative diff)
        to_positions = np.where(diff < 0)
        
        if len(from_positions[0]) > 0 and len(to_positions[0]) > 0:
            from_pos = (from_positions[0][0], from_positions[1][0])
            to_pos = (to_positions[0][0], to_positions[1][0])
            moves.append((from_pos, to_pos))
    
    return moves


def get_human_move(game_env: Checkers) -> np.ndarray:
    """
    Get move input from human player.
    
    Args:
        game_env: Checkers game environment.
        
    Returns:
        Selected next state.
    """
    legal_states = game_env.legal_next_states
    moves = states_to_moves(game_env.state, legal_states)
    
    while True:
        try:
            choice = input("\nEnter move number (or 'q' to quit): ")
            
            if choice.lower() == 'q':
                print("Thanks for playing!")
                sys.exit(0)
            
            move_idx = int(choice) - 1
            
            if 0 <= move_idx < len(legal_states):
                return legal_states[move_idx]
            else:
                print(f"Invalid choice. Please enter 1-{len(legal_states)}")
        except ValueError:
            print("Please enter a valid number")


def get_ai_move(
    game_env: Checkers,
    model: CheckersModel,
    config: Config,
    show_thinking: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Get move from AI using MCTS.
    
    Args:
        game_env: Checkers game environment.
        model: Neural network model.
        config: Configuration.
        show_thinking: Whether to show AI's thinking process.
        
    Returns:
        Tuple of (next_state, confidence).
    """
    mcts = MCTSPlayer(game_env, model, config, config.device)
    
    next_state, policy, value = mcts.get_action_probs(
        game_env.state,
        game_env.history,
        temperature=0,  # Deterministic for play
        add_noise=False
    )
    
    if show_thinking:
        # Show top moves
        legal_states = game_env.legal_next_states
        moves = states_to_moves(game_env.state, legal_states)
        
        print("\nAI thinking...")
        print(f"Position evaluation: {value:+.2f} (positive = AI advantage)")
        
        # Find which move was selected
        for idx, state in enumerate(legal_states):
            if np.array_equal(state[:14], next_state[:14]):
                print(f"Selected move: {moves[idx][0]} -> {moves[idx][1]}")
                break
    
    confidence = (value + 1) / 2  # Convert from [-1,1] to [0,1]
    return next_state, confidence


def play_game(
    model: CheckersModel,
    config: Config,
    human_player: Optional[int] = None,
    delay: float = 1.0
):
    """
    Play a game of Checkers.
    
    Args:
        model: Neural network model.
        config: Configuration.
        human_player: 0 for human as player 1, 1 for player 2, None for AI vs AI.
        delay: Delay between AI moves (seconds).
    """
    import time
    
    game_env = Checkers()
    
    print("\n" + "=" * 60)
    print("CHECKERS - PyTorch MCTS AI")
    print("=" * 60)
    
    if human_player is not None:
        human_marker = 'x' if human_player == 0 else 'o'
        ai_marker = 'o' if human_player == 0 else 'x'
        print(f"You are playing as: {human_marker} (Player {human_player + 1})")
        print(f"AI is playing as: {ai_marker}")
    else:
        print("AI vs AI demonstration")
    
    print("=" * 60)
    
    while not game_env.done:
        current_player = int(game_env.state[4, 0, 0])
        
        print_board(game_env, highlight_moves=(human_player == current_player))
        
        if human_player == current_player:
            # Human's turn
            next_state = get_human_move(game_env)
        else:
            # AI's turn
            print("\nAI is thinking...")
            next_state, confidence = get_ai_move(game_env, model, config)
            
            if human_player is None:
                time.sleep(delay)  # Add delay for AI vs AI visualization
        
        game_env.step(next_state)
        print("\n" + "-" * 40)
    
    # Game over
    print_board(game_env, highlight_moves=False)
    
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    
    if game_env.outcome == 'draw':
        print("The game is a draw!")
    elif game_env.outcome == 'player1_wins':
        if human_player == 0:
            print("Congratulations! You won!")
        elif human_player == 1:
            print("AI wins! Better luck next time.")
        else:
            print("Player 1 (x) wins!")
    else:
        if human_player == 1:
            print("Congratulations! You won!")
        elif human_player == 0:
            print("AI wins! Better luck next time.")
        else:
            print("Player 2 (o) wins!")
    
    print(f"Total moves: {game_env.move_count}")


def main():
    """Main function."""
    args = parse_args()
    
    print("Loading model...")
    model, config = load_model(args.model, args.device)
    config.num_simulations = args.simulations
    
    print(f"Model loaded: {config.num_res_blocks} blocks, {config.num_channels} channels")
    print(f"MCTS simulations: {config.num_simulations}")
    print(f"Device: {config.device}")
    
    # Determine game mode
    if args.ai_vs_ai:
        human_player = None
    elif args.human_first:
        human_player = 0
    else:
        human_player = 1  # Default: human plays second
    
    play_game(model, config, human_player, args.delay)
    
    # Ask to play again
    while True:
        again = input("\nPlay again? (y/n): ")
        if again.lower() == 'y':
            play_game(model, config, human_player, args.delay)
        else:
            print("Thanks for playing!")
            break


if __name__ == '__main__':
    main()
