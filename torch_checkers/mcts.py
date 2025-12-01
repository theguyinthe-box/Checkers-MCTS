"""
###############################################################################
# torch/mcts.py
#
# Monte Carlo Tree Search with neural network guidance for Checkers.
#
# This module implements MCTS following the AlphaZero algorithm:
# - Uses neural network for both policy prior and value estimation
# - PUCT selection criterion for balancing exploration/exploitation
# - Supports batched inference for GPU efficiency
# - Dirichlet noise for exploration during training
#
# The implementation is designed to work with the Checkers.py game logic
# while leveraging PyTorch for neural network operations.
#
###############################################################################
"""

import math
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
from copy import deepcopy
import sys
import os

# Add parent directory to path for Checkers import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    
    Each node represents a game state and stores:
    - Visit count (N): Number of times this node was visited
    - Total value (W): Sum of values from all visits
    - Prior probability (P): NN's prior probability for this action
    - Children: Dict mapping actions to child nodes
    
    Attributes:
        state: Game state (numpy array from Checkers.py).
        parent: Parent node or None for root.
        action_idx: Action index that led to this state (tuple of layer, x, y).
        prior_prob: Prior probability from neural network.
        visit_count: Number of times node was visited during search.
        total_value: Sum of backpropagated values.
        children: Dictionary mapping action indices to child nodes.
        is_terminal: Whether this is a terminal (game over) state.
        terminal_value: Value if terminal (-1, 0, or 1).
    """
    
    def __init__(
        self,
        state: np.ndarray,
        parent: Optional['MCTSNode'] = None,
        action_idx: Optional[Tuple[int, int, int]] = None,
        prior_prob: float = 0.0
    ):
        """
        Initialize MCTS node.
        
        Args:
            state: Game state from Checkers.py.
            parent: Parent node or None for root.
            action_idx: (layer, x, y) tuple identifying the action.
            prior_prob: Prior probability from neural network.
        """
        self.state = state
        self.parent = parent
        self.action_idx = action_idx
        self.prior_prob = prior_prob
        
        # Statistics
        self.visit_count = 0
        self.total_value = 0.0
        
        # Children
        self.children: Dict[Tuple[int, int, int], MCTSNode] = {}
        
        # Terminal state info
        self.is_terminal = False
        self.terminal_value = None
        
        # For tracking which player made the move to reach this state
        self.player = int(state[4, 0, 0])
    
    @property
    def mean_value(self) -> float:
        """
        Calculate mean value Q(s,a) for this node.
        
        Returns:
            Mean value, or 0 if never visited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def ucb_score(self) -> float:
        """
        Calculate UCB score for node selection.
        
        Note: This is typically calculated in select_child with parent info.
        """
        return self.mean_value
    
    def is_expanded(self) -> bool:
        """Check if node has been expanded (has children)."""
        return len(self.children) > 0
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (not expanded or terminal)."""
        return not self.is_expanded() or self.is_terminal


class MCTSPlayer:
    """
    MCTS player that uses a neural network for move selection.
    
    This class implements the full MCTS algorithm with neural network
    guidance as used in AlphaZero. It interfaces with the Checkers
    game environment and uses PyTorch for neural network inference.
    
    Key Features:
    - Batched neural network inference for GPU efficiency
    - PUCT selection criterion with configurable exploration
    - Dirichlet noise for exploration during training
    - Temperature-based move selection for training vs evaluation
    
    Args:
        game_env: Instance of Checkers game environment.
        model: PyTorch neural network model.
        config: Configuration object with MCTS parameters.
        device: PyTorch device for inference.
    
    Example:
        >>> from Checkers import Checkers
        >>> game = Checkers()
        >>> model = CheckersModel(config).to(device)
        >>> mcts = MCTSPlayer(game, model, config)
        >>> state, policy, value = mcts.get_action_probs(game.state, temperature=1.0)
    """
    
    def __init__(
        self,
        game_env,
        model: torch.nn.Module,
        config,
        device: str = 'cuda'
    ):
        """
        Initialize MCTS player.
        
        Args:
            game_env: Checkers game environment instance.
            model: Neural network model.
            config: Configuration with MCTS parameters.
            device: Device for neural network inference.
        """
        self.game_env = game_env
        self.model = model
        self.config = config
        self.device = device
        
        # MCTS parameters
        self.num_simulations = config.num_simulations
        self.cpuct = config.cpuct
        self.dirichlet_alpha = config.dirichlet_alpha
        self.dirichlet_epsilon = config.dirichlet_epsilon
    
    def get_action_probs(
        self,
        root_state: np.ndarray,
        history: List[np.ndarray],
        temperature: float = 1.0,
        add_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run MCTS and return action probabilities.
        
        This is the main interface method. It runs MCTS simulations from
        the root state and returns the improved policy from visit counts.
        
        Args:
            root_state: Current game state from Checkers.py.
            history: List of previous states (for legal move generation).
            temperature: Temperature for converting visits to probabilities.
                        Higher = more exploration, lower = more exploitation.
            add_noise: Whether to add Dirichlet noise to root priors.
        
        Returns:
            Tuple of:
                - best_state: Selected next state
                - policy_probs: Probability distribution over actions (8x8x8)
                - root_value: Value estimate of root state
        """
        # Create root node
        root = MCTSNode(root_state)
        
        # Get legal next states from game environment
        legal_states = self.game_env.get_legal_next_states(history)
        
        if not legal_states:
            # No legal moves (game over)
            return root_state, np.zeros((8, 8, 8)), 0.0
        
        # Expand root with neural network evaluation
        root_value = self._expand_node(root, legal_states, add_noise=add_noise)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree until leaf
            while node.is_expanded() and not node.is_terminal:
                node = self._select_child(node)
                search_path.append(node)
            
            # Get value for leaf node
            if node.is_terminal:
                value = node.terminal_value
            else:
                # Expand leaf node
                # Get legal moves for this state
                node_history = history.copy()
                for n in search_path[1:]:  # Skip root which is already in history
                    node_history.append(n.state)
                
                child_states = self.game_env.get_legal_next_states(node_history)
                
                if not child_states:
                    # Terminal state
                    done, outcome = self.game_env.determine_outcome(node_history)
                    node.is_terminal = True
                    node.terminal_value = self._outcome_to_value(outcome, node.player)
                    value = node.terminal_value
                else:
                    value = self._expand_node(node, child_states)
            
            # Backpropagation
            self._backpropagate(search_path, value)
        
        # Extract policy from visit counts
        policy_probs = self._get_policy_from_visits(root, temperature)
        
        # Select action
        if temperature == 0:
            # Deterministic: pick most visited
            best_child = max(root.children.values(), key=lambda c: c.visit_count)
        else:
            # Sample from distribution
            actions = list(root.children.keys())
            visits = [root.children[a].visit_count for a in actions]
            if temperature != 1.0:
                visits = [v ** (1/temperature) for v in visits]
            total = sum(visits)
            probs = [v/total for v in visits]
            idx = np.random.choice(len(actions), p=probs)
            best_child = root.children[actions[idx]]
        
        return best_child.state, policy_probs, root.mean_value
    
    def _expand_node(
        self,
        node: MCTSNode,
        legal_states: List[np.ndarray],
        add_noise: bool = False
    ) -> float:
        """
        Expand a node by adding children and evaluating with neural network.
        
        Args:
            node: Node to expand.
            legal_states: List of legal next states.
            add_noise: Whether to add Dirichlet noise to priors.
        
        Returns:
            Value estimate for the node's state.
        """
        # Prepare state for neural network
        state_tensor = self._state_to_tensor(node.state)
        
        # Get neural network prediction
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1)
        
        policy_probs = policy_probs.cpu().numpy().reshape(8, 8, 8)
        value = value.item()
        
        # Create action mask from legal states
        action_mask = np.zeros((8, 8, 8))
        action_to_state = {}
        
        for next_state in legal_states:
            # Extract action index from state (layer-6, x, y from channel 14)
            layer = int(next_state[14, 0, 0]) - 6
            x = int(next_state[14, 0, 1])
            y = int(next_state[14, 0, 2])
            action_idx = (layer, x, y)
            
            action_mask[layer, x, y] = 1
            action_to_state[action_idx] = next_state
        
        # Mask and normalize policy
        masked_policy = policy_probs * action_mask
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            # Uniform over legal moves
            masked_policy = action_mask / action_mask.sum()
        
        # Add Dirichlet noise for exploration
        if add_noise and len(action_to_state) > 0:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(action_to_state)
            )
            idx = 0
            for action_idx in action_to_state.keys():
                layer, x, y = action_idx
                masked_policy[layer, x, y] = (
                    (1 - self.dirichlet_epsilon) * masked_policy[layer, x, y] +
                    self.dirichlet_epsilon * noise[idx]
                )
                idx += 1
        
        # Create child nodes
        for action_idx, next_state in action_to_state.items():
            layer, x, y = action_idx
            prior = masked_policy[layer, x, y]
            child = MCTSNode(
                state=next_state,
                parent=node,
                action_idx=action_idx,
                prior_prob=prior
            )
            node.children[action_idx] = child
        
        return value
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select child node using PUCT criterion.
        
        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            node: Parent node to select from.
        
        Returns:
            Selected child node with highest PUCT score.
        """
        best_score = -float('inf')
        best_child = None
        
        sqrt_parent_visits = math.sqrt(node.visit_count)
        
        for child in node.children.values():
            # PUCT formula
            exploration = (
                self.cpuct * child.prior_prob * sqrt_parent_visits / 
                (1 + child.visit_count)
            )
            
            # Q value from child's perspective (negated for opponent)
            if child.player != node.player:
                q_value = -child.mean_value
            else:
                q_value = child.mean_value
            
            score = q_value + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """
        Backpropagate value through search path.
        
        The value is flipped when passing through nodes of the
        opposing player to maintain proper minimax semantics.
        
        Args:
            search_path: List of nodes from root to leaf.
            value: Value to backpropagate.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            
            # Value is from the perspective of the player who made the
            # move to reach this node. Flip sign for alternating players.
            if node.parent is not None and node.player != node.parent.player:
                value = -value
            
            node.total_value += value
    
    def _get_policy_from_visits(
        self,
        root: MCTSNode,
        temperature: float
    ) -> np.ndarray:
        """
        Convert visit counts to policy probabilities.
        
        Args:
            root: Root node with visit counts.
            temperature: Temperature for distribution.
        
        Returns:
            Policy probability array of shape (8, 8, 8).
        """
        policy = np.zeros((8, 8, 8))
        
        if temperature == 0:
            # One-hot for most visited
            max_visits = 0
            best_action = None
            for action_idx, child in root.children.items():
                if child.visit_count > max_visits:
                    max_visits = child.visit_count
                    best_action = action_idx
            if best_action:
                policy[best_action] = 1.0
        else:
            # Softmax with temperature
            visits = []
            actions = []
            for action_idx, child in root.children.items():
                actions.append(action_idx)
                visits.append(child.visit_count)
            
            if visits:
                visits = np.array(visits, dtype=np.float32)
                if temperature != 1.0:
                    visits = visits ** (1/temperature)
                probs = visits / visits.sum()
                
                for action_idx, prob in zip(actions, probs):
                    policy[action_idx] = prob
        
        return policy
    
    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """
        Convert numpy state to PyTorch tensor.
        
        Args:
            state: Game state array of shape (15, 8, 8).
        
        Returns:
            Tensor of shape (1, 14, 8, 8) on correct device.
        """
        # Use only first 14 channels
        tensor = torch.from_numpy(state[:14].copy()).float()
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def _outcome_to_value(self, outcome: str, player: int) -> float:
        """
        Convert game outcome string to value.
        
        Args:
            outcome: 'player1_wins', 'player2_wins', or 'draw'.
            player: Current player (0 or 1).
        
        Returns:
            Value in [-1, 1] from current player's perspective.
        """
        if outcome == 'draw':
            return 0.0
        elif outcome == 'player1_wins':
            return 1.0 if player == 0 else -1.0
        elif outcome == 'player2_wins':
            return 1.0 if player == 1 else -1.0
        return 0.0


def run_self_play_game(
    game_env,
    model: torch.nn.Module,
    config,
    device: str = 'cuda'
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Play a complete game of self-play and return training data.
    
    This function plays a full game using MCTS with the given neural
    network, collecting states, policies, and values for training.
    
    Args:
        game_env: Checkers game environment.
        model: Neural network model.
        config: Configuration object.
        device: Device for inference.
    
    Returns:
        List of (state, policy, value) tuples for training.
        The value is filled in after the game ends based on outcome.
    """
    mcts = MCTSPlayer(game_env, model, config, device)
    game_env.reset()
    
    experiences = []
    move_count = 0
    
    while not game_env.done and move_count < config.max_game_moves:
        # Determine temperature based on move count
        if move_count < config.temperature_drop_move:
            temperature = config.temperature
        else:
            temperature = config.final_temperature
        
        # Get action from MCTS
        next_state, policy, value = mcts.get_action_probs(
            game_env.state,
            game_env.history,
            temperature=temperature,
            add_noise=True  # Add noise during training
        )
        
        # Store experience
        experiences.append({
            'state': game_env.state.copy(),
            'policy': policy.copy(),
            'q_value': value,
            'player': int(game_env.state[4, 0, 0])
        })
        
        # Make move
        game_env.step(next_state)
        move_count += 1
    
    # Handle terminated games (no winner yet)
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
    
    # Convert outcome to values and create training data
    training_data = []
    for exp in experiences:
        # Value from player's perspective
        if outcome == 'draw':
            z_value = 0.0
        elif outcome == 'player1_wins':
            z_value = 1.0 if exp['player'] == 0 else -1.0
        else:  # player2_wins
            z_value = 1.0 if exp['player'] == 1 else -1.0
        
        training_data.append((
            exp['state'][:14],  # State without action channel
            exp['policy'],      # MCTS policy
            exp['q_value'],     # Q-value from MCTS
            z_value            # Actual game outcome
        ))
    
    return training_data, outcome, move_count
