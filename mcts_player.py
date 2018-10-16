"""This module containts the abstract class Player and some implementations."""
import sys

import time

import numpy as np
import multiprocessing as mp

from collections import defaultdict

from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, str_to_bitmask, translate_hand_cards, transform

from dragon_rider_player import RiderPlayer, policy_value_fn
from mcts import TreeNode, MCTS

from simulated_player import TIMEOUT_SECOND
from new_simulated_player import MonteCarloPlayer7
from strategy_play import random_choose


class MCTSPlayer(RiderPlayer):
    """AI player based on MCTS"""
    def __init__(self, verbose=False, c_puct=2):
        super(MCTSPlayer, self).__init__(verbose=verbose, c_puct=c_puct)


    def set_position(self, idx):
        super(MCTSPlayer, self).set_position(idx)

        self.mcts = [MCTS(policy_value_fn, self.position, self.c_puct) for _ in range(self.num_of_cpu)]


    def reset(self):
        super(RiderPlayer, self).reset()

        for idx in range(self.num_of_cpu):
            self.mcts[idx].update_with_move(-1)


    def see_played_trick(self, card, game):
        super(RiderPlayer, self).see_played_trick(card, game)

        card = tuple([SUIT_TO_INDEX[card.suit.__repr__()], NUM_TO_INDEX[card.rank.__repr__()]])

        is_found = []
        is_not_found = []
        for idx in range(self.num_of_cpu):
            is_reset = self.mcts[idx].update_with_move(card)

            if not is_reset:
                is_found.append(idx)
            else:
                is_not_found.append(idx)

        if is_found and is_not_found:
            for idx in is_not_found:
                self.mcts[idx] = self.mcts[np.random.choice(is_found)]

        if is_found:
            hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
                self.get_simple_game_info(game)

            pool = mp.Pool(processes=self.num_of_cpu)

            mul_result = [pool.apply_async(self.mcts[idx].get_move, 
                                           args=(hand_cards, 
                                                 remaining_cards, 
                                                 score_cards, 
                                                 init_trick, 
                                                 void_info, 
                                                 must_have, 
                                                 selection_func, 
                                                 game.trick_nr+1, 
                                                 game.is_heart_broken, 
                                                 game.expose_heart_ace, 
                                                 False,
                                                 0.11)) for idx in range(self.num_of_cpu)]

            results = [res.get() for res in mul_result]
            pool.close()


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND-0.05):
        stime = time.time()

        game.are_hearts_broken()

        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            self.get_simple_game_info(game)

        pool = mp.Pool(processes=self.num_of_cpu)

        mul_result = [pool.apply_async(self.mcts[idx].get_move, 
                                       args=(hand_cards, 
                                             remaining_cards, 
                                             score_cards, 
                                             init_trick, 
                                             void_info, 
                                             must_have, 
                                             selection_func, 
                                             game.trick_nr+1, 
                                             game.is_heart_broken, 
                                             game.expose_heart_ace, 
                                             False,
                                             simulation_time_limit)) for idx in range(self.num_of_cpu)]

        results = [res.get() for res in mul_result]

        cards = defaultdict(int)
        for idx, mcts in enumerate(results):
            for played_card, node in sorted(mcts._root._children.items(), key=lambda x: x[1]._n_visits):
                cards[played_card] += node._n_visits

            self.mcts[idx] = mcts

        for played_card, n_visits in sorted(cards.items(), key=lambda x: x[1]):
            played_card = transform(INDEX_TO_NUM[played_card[1]], INDEX_TO_SUIT[played_card[0]])
            self.say("played_card: {}, n_visits: {}", played_card, n_visits)

        pool.close()

        self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", (time.time()-stime), hand_cards, valid_cards, played_card)

        return played_card
