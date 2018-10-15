"""This module containts the abstract class Player and some implementations."""
import sys

import copy
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


class MCTSPlayer(RiderPlayer):
    """AI player based on MCTS"""
    def __init__(self, self_player_idx, verbose=False, c_puct=2):
        super(MCTSPlayer, self).__init__(self_player_idx=self_player_idx, verbose=verbose)

        self.mcts = [MCTS(policy_value_fn, self_player_idx, c_puct) for _ in range(self.num_of_cpu)]


    def reset(self):
        super(RiderPlayer, self).reset()

        for idx in range(self.num_of_cpu):
            self.mcts[idx].update_with_move(-1)


    def see_played_trick(self, card):
        super(RiderPlayer, self).see_played_trick(card)

        card = tuple([SUIT_TO_INDEX[card.suit.__repr__()], NUM_TO_INDEX[card.rank.__repr__()]])

        for idx in range(self.num_of_cpu):
            self.mcts[idx].update_with_move(card)


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        game.are_hearts_broken()

        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            self.get_simple_game_info(copy.deepcopy(game))

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
                                             False)) for idx in range(self.num_of_cpu)]

        results = [res.get() for res in mul_result]

        cards = defaultdict(int)
        for sub_results in results:
            for played_card, info in sub_results.items():
                cards[played_card] += info[0]

        for played_card, n_visits in sorted(cards.items(), key=lambda x: x[1]):
            played_card = transform(INDEX_TO_NUM[played_card[1]], INDEX_TO_SUIT[played_card[0]])
            self.say("played_card: {}, n_visits: {}", played_card, n_visits)

        pool.close()

        self.say("Hand card: {}, Validated card: {}, Picked card: {}", hand_cards, valid_cards, played_card)

        return played_card
