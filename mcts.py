"""This module containts the abstract class Player and some implementations."""
import os
import sys

import copy
import time

from random import shuffle, randint, choice
from statistics import mean
from math import sqrt

from collections import defaultdict

from card import Suit, Rank, Card, Deck
from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, str_to_bitmask, translate_hand_cards, transform

from simulated_player import TIMEOUT_SECOND

from step_game import StepGame
from redistribute_cards import redistribute_cards


OUT_FILE = None

def say(message, *formatargs):
    global OUT_FILE

    message = message.format(*formatargs)
    if os.path.exists("/log"):
        if OUT_FILE is None: OUT_FILE = open("/log/mcts.log", "a", 1)

        OUT_FILE.write("{}\n".format(message))
    else:
        print(message)


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
        #print("for {} to extend {}".format(player_idx, action_priors))
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, self._self_player_idx, player_idx)
            else:
                self._children[action]._player_idx = player_idx
                self._children[action]._P = prob


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
        self._u = (c_puct * sqrt(self._parent._n_visits) / (1 + self._n_visits))

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
        self.start_node = self._root

        self._policy = policy_value_fn
        self._c_puct = c_puct

        self.ratio_reset = [0, 0]


    def _playout(self, trick_nr, state, selection_func):
        #node = self._root
        node = self.start_node
        while not state.is_finished:
            if node.is_leaf():
                break

            is_all_traverse = True

            valid_cards = state.get_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1)
            valid_moves = translate_hand_cards(valid_cards, is_bitmask=True)
            for card in valid_moves:
                if card not in node._children or node._children[card]._P == 0:
                    is_all_traverse = False

                    break

            if not is_all_traverse:
                break

            is_found = False
            for played_card, n in node.select(self._c_puct):
                suit, rank = played_card[0], played_card[1]

                #if n._player_idx == 3 and trick_nr > 4:
                #    print(is_all_traverse, valid_moves, n._player_idx, played_card, n._P)

                if valid_cards.get(suit, 0) & rank:
                    state.step(trick_nr, selection_func, played_card)
                    is_found = True

                    node = n

                    break

            if not is_found:
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
                 valid_cards,
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

        simulation_cards = redistribute_cards(randint(0, 128), 
                                              self._self_player_idx, 
                                              copy.deepcopy(hand_cards), 
                                              init_trick[-1][1], 
                                              remaining_cards, 
                                              must_have, 
                                              void_info)

        valid_cards = str_to_bitmask(valid_cards)

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
                                  hand_cards=simulation_card,
                                  void_info=void_info,
                                  score_cards=copy.deepcopy(score_cards), 
                                  is_hearts_broken=is_heart_broken, 
                                  expose_hearts_ace=expose_heart_ace, 
                                  tricks=copy.deepcopy(init_trick))

                    if len(init_trick[-1][1]) == 4:
                        sm.post_round_end()

                    selection_func = [choice(selection_func) for _ in range(4)]
                    self._playout(trick_nr, sm, selection_func)
                except Exception as e:
                    for player_idx, cards in enumerate(simulation_card):
                        say("player-{}'s hand_cards is {}", player_idx, cards)

                    import traceback
                    traceback.print_exc()


            if time.time()-stime > simulation_time_limit:
                break

        if is_only_played_card:
            if is_print:
                for k, v in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                    if v._n_visits > 0 and valid_cards.get(k[0], 0) & k[1]:
                        say("{}: {} times, percentage: {:.4f}%", transform(INDEX_TO_NUM[k[1]], INDEX_TO_SUIT[k[0]]), v._n_visits, v._P*100)

                    if v._P == 0:
                        break

            for played_card, node in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                if node._n_visits > 0 and node._P > 0 and valid_cards.get(played_card[0], 0) & played_card[1]:
                    return played_card
        else:
            results = {}
            for played_card, node in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                if node._P > 0 and node._n_visits > 0 and valid_cards.get(played_card[0], 0) & played_card[1]:
                    results.setdefault(played_card, 0)
                    results[played_card] = node._n_visits

            return results


    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """

        if last_move in self.start_node._children:
            self.start_node = self.start_node._children[last_move]
            self.ratio_reset[0] += 1

            #print("reset the root to be {}".format(last_move))
        else:
            self.start_node = TreeNode(None, 1.0, self._self_player_idx, None)

            """
            if self.start_node._parent is None:
                self.start_node = TreeNode(None, 1.0, self._self_player_idx, None)
            else:
                mean_prob = mean([node._P for node in self.start_node._parent._children.values()])
                self.start_node._parent.expand(self._self_player_idx, [[last_move, mean_prob]])
            """

            self.ratio_reset[1] += 1

            say("reset the root because {} is NOT found", last_move)
            say("the reset ratio: {}", self.ratio_reset)


    def print_tree(self, node=None, card=None, depth=0):
        node = self._root if node is None else node

        if node._parent:
            card_str = transform(INDEX_TO_NUM[card[1]], INDEX_TO_SUIT[card[0]])

            say("{} {} {} {} {} {}", "****"*depth, card_str, node, node._n_visits, node.get_value(self._c_puct), node._parent)

        for card, children in node._children.items():
            self.print_tree(children, card, depth+1)


    def __str__(self):
        return "MCTS"
