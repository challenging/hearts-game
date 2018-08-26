"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np
import multiprocessing as mp

from collections import defaultdict
from random import shuffle, choice

from card import Suit, Rank, Card, Deck
from rules import is_card_valid

from simulated_player import MonteCarloPlayer2
from player import SimplePlayer, StupidPlayer

TIMEOUT_SECOND = 0.9
COUNT_CPU = mp.cpu_count()


from operator import itemgetter


def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(state):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS

    #action_probs = np.ones(len(board.availables))/len(board.availables)

    current_players = state.players[state.current_player_idx]
    valid_moves = current_players.get_valid_cards(state._player_hands[state.current_player_idx], state.trick, state.trick_nr, state.is_heart_broken)
    #print(state.current_player_idx, state._player_hands[state.current_player_idx], valid_moves)

    action_probs = np.ones(len(valid_moves)) / len(valid_moves)

    return zip(valid_moves, action_probs), 0


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p


    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

        #print(self._parent, self, self._children)


    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))


    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            --perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits


    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)

        self.update(leaf_value)


    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))

        return self._Q + self._u


    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}


    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.

        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout


    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break

            # Greedily select next move.
            action, node = node.select(self._c_puct)

            state.step(action)

        action_probs, _ = self._policy(state)

        # Check for end of game
        winners = state.get_game_winners()
        if not winners:
            node.expand(action_probs)

        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        #print("leaf_value", leaf_value)

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)


    def _evaluate_rollout(self, state):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """

        winners = state.get_game_winners()
        while not winners:
            state.step()
            winners = state.get_game_winners()

        #return 1 if state.current_player_idx in winners else -1

        return -state.player_scores[state.current_player_idx]


    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action
        """

        state_copy = copy.deepcopy(state)
        seen_cards = state.players[0].seen_cards
        for n in range(self._n_playout):
            hand_cards = state_copy._player_hands[state.current_player_idx]
            remaining_cards = state.players[state.current_player_idx].get_remaining_cards(state._player_hands[state.current_player_idx])
            state.players[state.current_player_idx].redistribute_cards(state, remaining_cards[:])

            state_copy.verbose = False
            state_copy.players = [StupidPlayer() for idx in range(4)]
            for player in state_copy.players:
                player.seen_cards = copy.deepcopy(seen_cards)

            self._playout(copy.deepcopy(state_copy))

        for played_card, node in sorted(self._root._children.items(), key=lambda x: -x[1].get_value(self._c_puct)):
            print(played_card, node._n_visits, node.get_value(self._c_puct))

        return max(self._root._children.items(), key=lambda act_node: act_node[1].get_value(self._c_puct))[0]


    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


    def __str__(self):
        return "MCTS"


class MCTSPlayer(MonteCarloPlayer2):
    """AI player based on MCTS"""
    def __init__(self, verbose=False, c_puct=5, n_playout=2048):
        super(MCTSPlayer, self).__init__(verbose)

        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)


    def reset(self):
        super(MCTSPlayer, self).reset()

        self.mcts.update_with_move(-1)


    def play_card(self, hand_cards, game, simulation_time_limit=TIMEOUT_SECOND):
        valid_cards = self.get_valid_cards(hand_cards, game.trick, game.trick_nr, game.is_heart_broken)

        if len(valid_cards) > 1:
            played_card = self.mcts.get_move(copy.deepcopy(game))
            print("pick card", played_card)
            self.mcts.update_with_move(-1)
        else:
            played_card = valid_cards[0]

            if self.verbose:
                print("don't need simulation, can only play {} card".format(played_card))

        return played_card
