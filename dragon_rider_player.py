"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np
import multiprocessing as mp

from card import Suit, Rank, Card, Deck

from simple_game import SimpleGame
from simulated_player import MonteCarloPlayer5
from player import StupidPlayer


TIMEOUT_SECOND = 0.93
COUNT_CPU = 1#mp.cpu_count()


def policy_value_fn(state):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""

    current_players = state.players[state.current_player_idx]
    valid_moves = current_players.get_valid_cards(state._player_hands[state.current_player_idx], state)

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
                self._children[(player_idx, action)] = TreeNode(self, prob, self._self_player_idx, player_idx)


    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """

        return sorted(self._children.items(),
                   key=lambda act_node: -act_node[1].get_value(c_puct))


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
            is_all_traverse = True
            valid_cards = state.players[state.current_player_idx].get_valid_cards(state._player_hands[state.current_player_idx], state)
            for card in valid_cards:
                if (state.current_player_idx, card) not in node._children:
                    is_all_traverse = False

                    break

            if not is_all_traverse:
                break

            is_valid = False
            for action, node in node.select(self._c_puct):
                player_idx, played_card = action

                if played_card in valid_cards:
                    state.step(played_card)
                    is_valid = True

                    break

            if not is_valid:
                break


        action_probs, _ = self._policy(state)
        action_probs = list(action_probs)

        # Check for end of game
        winners = state.get_game_winners()
        if not winners:
            node.expand(state.current_player_idx, action_probs)

        # Evaluate the leaf node by random rollout
        #leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        #node.update_recursive(-leaf_value)

        scores = self._evaluate_rollout(state)
        node.update_recursive(scores)


    def _evaluate_rollout(self, state):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """

        winners = state.get_game_winners()
        while not winners:
            state.step()
            winners = state.get_game_winners()

        scores = state.player_scores
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


    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action
        """

        state_copy = copy.deepcopy(state)
        seen_cards = state.players[0].seen_cards

        stime = time.time()
        while time.time()-stime < TIMEOUT_SECOND:
            remaining_cards = state.players[state.current_player_idx].get_remaining_cards(state._player_hands[state.current_player_idx])
            state_copy = state.players[state.current_player_idx].redistribute_cards(state_copy, remaining_cards[:])

            state_copy.verbose = False
            state_copy.players = [StupidPlayer() for idx in range(4)]
            for player in state_copy.players:
                player.seen_cards = copy.deepcopy(seen_cards)

            self._playout(copy.deepcopy(state_copy))

        for (player_idx, played_card), node in sorted(self._root._children.items(), key=lambda x: x[1]._n_visits):
            print(player_idx, played_card, node.get_value(self._c_puct), node._n_visits)

        #return self._root
        return played_card, node.get_value(self._c_puct)


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

        #print(node, card, depth, node._children)
        if node._parent:
            print("  "*depth, card, "[{}]".format(node._player_idx), node, node._n_visits, node.get_value(self._c_puct))

        for card, children in node._children.items():
            self.print_tree(children, card, depth+1)

    def __str__(self):
        return "MCTS"


class DragonRiderPlayer(MonteCarloPlayer5):
    """AI player based on MCTS"""
    def __init__(self, self_player_idx, verbose=False, c_puct=2):
        super(DragonRiderPlayer, self).__init__(verbose)

        self.mcts = MCTS(policy_value_fn, self_player_idx, c_puct)


    def reset(self):
        super(DragonRiderPlayer, self).reset()

        self.mcts.update_with_move(-1)


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        game.are_hearts_broken()

        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        if len(valid_cards) > 1:
            """
            pool = mp.Pool(processes=self.num_of_cpu)

            mul_result = [pool.apply_async(self.mcts.get_move, args=(game,)) for _ in range(COUNT_CPU)]
            results = [res.get() for res in mul_result]

            cards = {}
            for root in results:
                for (player_idx, played_card), node in root._children.items():
                    cards.setdefault(played_card, 0)
                    cards[played_card] += node._n_visits
                    #print("---->", played_card, node._n_visits)

            for played_card, n_visits in sorted(cards.items(), key=lambda x: x[1]):
                self.say("played_card: {}, n_visits: {}", played_card, n_visits)

            pool.close()
            """

            played_card, _ = self.mcts.get_move(copy.deepcopy(game))

            self.say("Hand card: {}, Validated card: {}, Picked card: {}", hand_cards, valid_cards, played_card)
            #played_card = played_card[0]

            self.mcts.print_tree()
            self.mcts.update_with_move(-1)
        else:
            played_card = self.no_choice(valid_cards[0])

        return played_card
