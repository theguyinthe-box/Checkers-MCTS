#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# torch_checkers/random_player.py
#
# Random Legal Move player for Checkers benchmarking.
#
# This module implements a simple player that selects moves uniformly at
# random from all legal moves. It serves as a baseline for evaluating
# trained neural network models.
#
# Usage:
#   from torch_checkers.random_player import RandomPlayer
#   player = RandomPlayer(game_env)
#   next_state = player.get_action(game_env.state, game_env.history)
#
###############################################################################
"""

import numpy as np
from typing import Tuple, List, Optional
import sys
import os

# Add parent directory to path for Checkers import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RandomPlayer:
    """
    Random Legal Move player for Checkers.
    
    This player selects moves uniformly at random from all legal moves.
    It serves as a baseline for benchmarking trained models.
    
    Args:
        game_env: Instance of Checkers game environment.
        seed: Optional random seed for reproducibility.
    
    Example:
        >>> from Checkers import Checkers
        >>> from torch_checkers.random_player import RandomPlayer
        >>> game = Checkers()
        >>> player = RandomPlayer(game)
        >>> next_state, policy, _ = player.get_action(game.state, game.history)
    """
    
    def __init__(self, game_env, seed: Optional[int] = None):
        """
        Initialize the random player.
        
        Args:
            game_env: Checkers game environment instance.
            seed: Optional random seed for reproducibility.
        """
        self.game_env = game_env
        if seed is not None:
            np.random.seed(seed)
    
    def get_action(
        self,
        state: np.ndarray,
        history: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get a random legal action.
        
        Args:
            state: Current game state from Checkers.py.
            history: List of previous states (for legal move generation).
        
        Returns:
            Tuple of:
                - best_state: Selected next state (randomly chosen)
                - policy_probs: Uniform probability distribution over legal actions (8x8x8)
                - value: Always returns 0 (no value estimation)
        """
        # Get legal next states from game environment
        legal_states = self.game_env.get_legal_next_states(history)
        
        if not legal_states:
            # No legal moves (game over)
            return state, np.zeros((8, 8, 8)), 0.0
        
        # Create uniform policy over legal moves
        policy_probs = np.zeros((8, 8, 8))
        num_legal_moves = len(legal_states)
        prob_per_move = 1.0 / num_legal_moves
        
        for next_state in legal_states:
            # Extract action index from state (layer-6, x, y from channel 14)
            layer = int(next_state[14, 0, 0]) - 6
            x = int(next_state[14, 0, 1])
            y = int(next_state[14, 0, 2])
            policy_probs[layer, x, y] = prob_per_move
        
        # Select random move
        random_idx = np.random.randint(0, num_legal_moves)
        selected_state = legal_states[random_idx]
        
        return selected_state, policy_probs, 0.0
    
    def get_action_probs(
        self,
        state: np.ndarray,
        history: List[np.ndarray],
        temperature: float = 1.0,
        add_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get action probabilities (compatible interface with MCTSPlayer).
        
        This method provides the same interface as MCTSPlayer for easy
        substitution during evaluation.
        
        Args:
            state: Current game state.
            history: List of previous states.
            temperature: Ignored (always uses uniform random).
            add_noise: Ignored (no noise needed for random policy).
        
        Returns:
            Tuple of (selected_state, policy_probs, value).
        """
        return self.get_action(state, history)


def play_random_game(game_env) -> Tuple[str, int]:
    """
    Play a complete game with two random players.
    
    This function is useful for testing and generating baseline statistics.
    
    Args:
        game_env: Checkers game environment instance.
    
    Returns:
        Tuple of (outcome, move_count).
    """
    game_env.reset()
    player = RandomPlayer(game_env)
    
    move_count = 0
    max_moves = 200  # Prevent infinite games
    
    while not game_env.done and move_count < max_moves:
        next_state, _, _ = player.get_action(game_env.state, game_env.history)
        game_env.step(next_state)
        move_count += 1
    
    # Determine outcome
    if not game_env.done:
        # Game was terminated, determine winner by piece count
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


if __name__ == '__main__':
    # Quick test of random player
    from Checkers import Checkers
    
    print("Testing RandomPlayer...")
    
    game = Checkers()
    outcomes = {'player1_wins': 0, 'player2_wins': 0, 'draw': 0}
    total_moves = 0
    num_games = 10
    
    for i in range(num_games):
        game.reset()
        outcome, moves = play_random_game(game)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total_moves += moves
        print(f"Game {i+1}: {outcome} in {moves} moves")
    
    print(f"\nResults after {num_games} games:")
    print(f"  Player 1 wins: {outcomes['player1_wins']}")
    print(f"  Player 2 wins: {outcomes['player2_wins']}")
    print(f"  Draws: {outcomes['draw']}")
    print(f"  Avg moves: {total_moves / num_games:.1f}")
