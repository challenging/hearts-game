"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np

from collections import defaultdict

from card import Suit, Rank, Card, Deck
from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, str_to_bitmask, translate_hand_cards, transform

from simulated_player import TIMEOUT_SECOND

from simple_complex_game import SimpleGame
from redistribute_cards import redistribute_cards


def policy_value_fn(trick_nr, state):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""

    valid_cards = state.get_myself_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1)
    valid_moves = translate_hand_cards(valid_cards, is_bitmask=True)

    action_probs = np.ones(len(valid_moves)) / len(valid_moves)

    return zip(valid_moves, action_probs), 0


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

        return sorted(self._children.items(), key=lambda act_node: -act_node[1].get_value(c_puct))


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


    def _playout(self, trick_nr, state, selection_func):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break

            # Greedily select next move.
            is_all_traverse = True

            valid_cards = state.get_myself_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1)
            valid_moves = translate_hand_cards(valid_cards, is_bitmask=True)
            for card in valid_moves:
                if card not in node._children:
                    is_all_traverse = False

                    break

            if not is_all_traverse:
                break

            is_valid = False
            for played_card, node in node.select(self._c_puct):
                suit, rank = played_card[0], played_card[1]
                if valid_cards.get(suit, 0) & rank:
                    state.step(trick_nr, selection_func, played_card)
                    is_valid = True

                    break

            if not is_valid:
                break

        action_probs, _ = self._policy(trick_nr, state)
        action_probs = list(action_probs)

        # Check for end of game
        if not state.is_finished:
            node.expand(state.start_pos, action_probs)

        # Evaluate the leaf node by random rollout
        #leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        #node.update_recursive(-leaf_value)

        scores = self._evaluate_rollout(trick_nr, state, selection_func)
        node.update_recursive(scores)


    def _evaluate_rollout(self, trick_nr, state, selection_func):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """

        while not state.is_finished:
            state.step(trick_nr, selection_func)

        scores, _ = state.score()
        rating = [0, 0, 0, 0]

        info = zip(range(4), scores)
        pre_score, pre_rating, sum_score = None, None, np.array(scores)/np.sum(scores)
        for rating_idx, (player_idx, score) in enumerate(sorted(info, key=lambda x: -x[1])):
            tmp_rating = rating_idx
            if pre_score is not None:
                if score == pre_score:
                    tmp_rating = pre_rating

            rating[player_idx] = (tmp_rating/4 + (1-sum_score[player_idx]))/2

            pre_score = score
            pre_rating = tmp_rating

        return rating


    def get_move(self, hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func, trick_nr, is_heart_broken, expose_heart_ace, is_only_played_card=False):
        stime = time.time()

        simulation_cards = redistribute_cards(int(time.time()*1000), 
                                              self._self_player_idx, 
                                              copy.deepcopy(hand_cards), 
                                              init_trick[-1][1], 
                                              remaining_cards, 
                                              must_have, 
                                              void_info)

        for simulation_card in simulation_cards:
            for player_idx, cards in enumerate(simulation_card):
                simulation_card[player_idx] = str_to_bitmask(cards)

            sm = SimpleGame(position=self._self_player_idx, 
                            hand_cards=simulation_card, 
                            void_info=copy.deepcopy(void_info),
                            score_cards=copy.deepcopy(score_cards), 
                            is_hearts_broken=is_heart_broken, 
                            expose_hearts_ace=expose_heart_ace, 
                            tricks=copy.deepcopy(init_trick))

            self._playout(trick_nr, sm, selection_func)

            if time.time()-stime > TIMEOUT_SECOND:
                break

        results = defaultdict(list)
        total_n_visits = 0
        for played_card, node in sorted(self._root._children.items(), key=lambda x: x[1]._n_visits):
            print("---->", transform(INDEX_TO_NUM[played_card[1]], INDEX_TO_SUIT[played_card[0]]), node.get_value(self._c_puct), node._n_visits)

            results[played_card] = [node._n_visits, node.get_value(self._c_puct)]
            total_n_visits += node._n_visits

        for info in results.values():
            info[0] /= total_n_visits

        if is_only_played_card:
            return played_card
        else:
            return results


    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """

        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
            self._player_idx = None

            #print("reset the root to be {}".format(last_move))
        else:
            self._root = TreeNode(None, 1.0, self._self_player_idx, None)

            #print("reset the root because {} is NOT found".format(last_move))


    def print_tree(self, node=None, card=None, depth=0):
        node = self._root if node is None else node

        if node._parent:
            card_str = transform(INDEX_TO_NUM[card[1]], INDEX_TO_SUIT[card[0]])
            print("  "*depth, card_str, card, "[{}]".format(node._player_idx), node, node._n_visits, node.get_value(self._c_puct))

        for card, children in node._children.items():
            self.print_tree(children, card, depth+1)


    def __str__(self):
        return "MCTS"
