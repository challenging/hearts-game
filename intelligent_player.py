"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np
import multiprocessing as mp

from card import Suit, Rank, Card, Deck

from simulated_player import MonteCarloPlayer6
from player import StupidPlayer

TIMEOUT_SECOND = 0.93

from operator import itemgetter


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)

    return probs


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, self_player_idx, player_idx):
        self._parent = parent

        self._self_player_idx = self_player_idx
        self._player_idx = player_idx

        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p


    def expand(self, player_idx, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, self._self_player_idx, player_idx)


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
        self._Q += (leaf_value - self._Q) / self._n_visits


    def update_recursive(self, scores):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(scores)

        if self._player_idx is None:
            self._n_visits += 1
        else:
            self.update(scores[self._player_idx])#*(1 if self._self_player_idx == self._player_idx else -1))


    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))

        return self._Q + self._u


    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}


    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, self_player_idx, c_puct=5):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.

        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """

        self._self_player_idx = self_player_idx
        self._root = TreeNode(None, 1.0, self_player_idx, None)

        self._policy = policy_value_fn
        self._c_puct = c_puct


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

        action_probs, rating = self._policy(state)

        # Check for end of game.
        winners = state.get_game_winners()
        if not winners:
            node.expand(state.current_player_idx, action_probs)
        else:
            rating = [0, 0, 0, 0]

            info = zip(range(4), state.player_scores)
            pre_score, pre_rating, max_score = None, None, np.array(state.player_scores)/np.max(state.player_scores)
            for rating_idx, (player_idx, score) in enumerate(sorted(info, key=lambda x: x[1])):
                tmp_rating = rating_idx
                if pre_score is not None:
                    if score == pre_score:
                        tmp_rating = pre_rating


                rating[player_idx] = ((4-tmp_rating) + (1-max_score[player_idx]))/5

                pre_score = score
                pre_rating = tmp_rating

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(rating)


    def get_move_probs(self, state):
        state_copy = copy.deepcopy(state)
        seen_cards = state.players[0].seen_cards

        stime = time.time()
        while time.time()-stime < TIMEOUT_SECOND:
            hand_cards = state_copy._player_hands[state.current_player_idx]
            remaining_cards = state.players[state.current_player_idx].get_remaining_cards(state._player_hands[state.current_player_idx])
            state.players[state.current_player_idx].redistribute_cards(state, remaining_cards[:])

            state_copy.verbose = False
            state_copy.players = [StupidPlayer() for idx in range(4)]
            for player in state_copy.players:
                player.seen_cards = copy.deepcopy(seen_cards)

            self._playout(copy.deepcopy(state_copy))

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))


        return acts, act_probs


    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """

        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
            self._player_idx = None
        else:
            self._root = TreeNode(None, 1.0, self._self_player_idx, None)


    def print_tree(self, node=None, card=None, depth=0):
        node = self._root if node is None else node

        if node._children:
            if node._parent:
                print("\t"*depth, card, node.get_value(self._c_puct))
            for card, children in node._children.items():
                self.print_tree(children, card, depth+1)
        else:
            print("\t"*depth, card, node)

    def __str__(self):
        return "MCTS"


class IntelligentPlayer(MonteCarloPlayer6):
    """AI player based on MCTS"""
    def __init__(self, policy_value_fn, self_player_idx, is_selfplay, verbose=False, c_puct=2):
        super(IntelligentPlayer, self).__init__(verbose)

        self.mcts = MCTS(policy_value_fn, self_player_idx, c_puct)
        self.set_selfplay(is_selfplay)


    def set_selfplay(self, is_selfplay):
        self._is_selfplay = is_selfplay


    def reset(self):
        super(IntelligentPlayer, self).reset()

        self.mcts.update_with_move(-1)


    def play_card(self, game, temp=1e-3, return_prob=0):
        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        if len(valid_cards) > 1:
            #played_card = self.mcts.get_move(copy.deepcopy(game))
            #self.mcts.print_tree(depth=-1)

            #print("pick card", played_card, hand_cards, valid_cards)

            #self.mcts.update_with_move(-1)

            played_card = None
            acts, probs = self.mcts.get_move_probs(game, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                played_card = np.random.choice(acts,p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                # update the root node and reuse the search tree
                self.mcts.update_with_move(played_card)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                played_card = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
        else:
            played_card = valid_cards[0]

            self.sat("don't need simulation, can only play {} card", played_card)

        return played_card
