"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

#import numpy as np
from random import shuffle, randint
from statistics import mean
from math import sqrt

from collections import defaultdict

from card import Suit, Rank, Card, Deck
from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, str_to_bitmask, translate_hand_cards, transform

from simulated_player import TIMEOUT_SECOND

from step_game import StepGame
from redistribute_cards import redistribute_cards


def policy_value_fn(trick_nr, state):
    results = state.get_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1, is_playout=False)

    return results, 0


class TreeNode(object):
    def __init__(self, parent, prior_p, self_player_idx, player_idx):
        self._parent = parent

        self._self_player_idx = self_player_idx
        self._player_idx = player_idx

        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p


    def expand(self, player_idx, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, self._self_player_idx, player_idx)


    def select(self, c_puct):
        for card, node in sorted(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)*-1):
            if node._P > 0:
                yield card, node


    def update(self, leaf_value):
        self._n_visits += 1

        self._Q += leaf_value


    def update_recursive(self, scores):
        if self._parent:
            self._parent.update_recursive(scores)

        if self._player_idx is None:
            self._n_visits += 1
        else:
            v = scores[self._self_player_idx]*(1 if self._self_player_idx == self._player_idx else -1)
            self.update(v)


    def get_value(self, c_puct):
        self._u = (c_puct * self._P * sqrt(self._parent._n_visits) / (1 + self._n_visits))

        return self._Q/(1e-16+self._n_visits) + self._u


    def is_leaf(self):
        return self._children == {}


    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, self_player_idx, c_puct=5):
        self._self_player_idx = self_player_idx
        self._root = TreeNode(None, 1.0, self_player_idx, None)

        self._policy = policy_value_fn
        self._c_puct = c_puct


    def _playout(self, trick_nr, state, selection_func):
        node = self._root
        while not state.is_finished:
            if node.is_leaf():
                break

            is_all_traverse = True

            valid_cards = state.get_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1)
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

                #if node._player_idx == 3: print(node._player_idx, node._P, node._n_visits, node.get_value(self._c_puct))
                if valid_cards.get(suit, 0) & rank:
                    #if node._player_idx == 3: print("---->", node._player_idx, node._P, node._n_visits, node.get_value(self._c_puct))
                    state.step(trick_nr, selection_func, played_card)
                    is_valid = True

                    break

            if not is_valid:
                break

        # Check for end of game
        if not state.is_finished:
            action_probs, _ = self._policy(trick_nr, state)
            node.expand(state.start_pos, action_probs)

        scores = self._evaluate_rollout(trick_nr, state, selection_func)
        node.update_recursive(scores)


    def _evaluate_rollout(self, trick_nr, state, selection_func):
        while not state.is_finished:
            state.step(trick_nr, selection_func)

        scores, _ = state.score()
        sum_score = sum(scores)

        #return [1-(score/sum_score) for score in scores]

        rating = [0, 0, 0, 0]

        info = zip(range(4), scores)
        pre_score, pre_rating = None, None
        for rating_idx, (player_idx, score) in enumerate(sorted(info, key=lambda x: -x[1])):
            tmp_rating = rating_idx
            if pre_score is not None:
                if score == pre_score:
                    tmp_rating = pre_rating

            rating[player_idx] = tmp_rating/4 + (1-score/sum_score)

            pre_score = score
            pre_rating = tmp_rating

        return rating


    def get_move(self, 
                 hand_cards, 
                 remaining_cards, 
                 score_cards, 
                 init_trick, 
                 void_info, 
                 must_have, 
                 selection_func, 
                 trick_nr, 
                 is_heart_broken, 
                 expose_heart_ace, 
                 is_only_played_card=False, 
                 simulation_time_limit=TIMEOUT_SECOND-0.1,
                 is_print=False):

        stime = time.time()

        simulation_cards = redistribute_cards(randint(0, 256), 
                                              self._self_player_idx, 
                                              copy.deepcopy(hand_cards), 
                                              init_trick[-1][1], 
                                              remaining_cards, 
                                              must_have, 
                                              void_info)

        for simulation_card in simulation_cards:
            max_len_cards, min_min_cards = -1, 99
            for player_idx, cards in enumerate(simulation_card):
                simulation_card[player_idx] = str_to_bitmask(cards)

                if len(cards) > max_len_cards:
                    max_len_cards = len(cards)

                if len(cards) < min_min_cards:
                    min_min_cards = len(cards)

            if max_len_cards-min_min_cards < 2:
                try:
                    sm = StepGame(trick_nr,
                                  position=self._self_player_idx, 
                                  hand_cards=copy.deepcopy(simulation_card), 
                                  void_info=copy.deepcopy(void_info),
                                  score_cards=copy.deepcopy(score_cards), 
                                  is_hearts_broken=is_heart_broken, 
                                  expose_hearts_ace=expose_heart_ace, 
                                  tricks=copy.deepcopy(init_trick))

                    #shuffle(selection_func)
                    self._playout(trick_nr, sm, selection_func)
                except Exception as e:
                    for player_idx, cards in enumerate(simulation_card):
                        print("player-{}'s hand_cards is {}".format(player_idx, cards, simulation_card[player_idx]))

                    raise

            if time.time()-stime > simulation_time_limit:
                break

        if is_only_played_card:
            if is_print:
                for k, v in sorted(self._root._children.items(), key=lambda x: -x[1]._n_visits):
                    if v._n_visits > 0: print(transform(INDEX_TO_NUM[k[1]], INDEX_TO_SUIT[k[0]]), v._n_visits)

            return sorted(self._root._children.items(), key=lambda x: -x[1]._n_visits)[0][0]
        else:
            return self


    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """

        is_reset = True
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
            self._player_idx = None

            is_reset = False

            #print("reset the root to be {}".format(last_move))
        else:
            self._root = TreeNode(None, 1.0, self._self_player_idx, None)

            print("reset the root because {} is NOT found".format(last_move))

        return is_reset


    def print_tree(self, node=None, card=None, depth=0):
        node = self._root if node is None else node

        if node._parent:
            card_str = transform(INDEX_TO_NUM[card[1]], INDEX_TO_SUIT[card[0]])

            print("****"*depth, card_str, node, node._n_visits, node.get_value(self._c_puct), node._parent)

        for card, children in node._children.items():
            self.print_tree(children, card, depth+1)


    def __str__(self):
        return "MCTS"
