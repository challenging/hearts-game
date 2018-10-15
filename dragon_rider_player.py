"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np

from collections import defaultdict

from card import Deck, Card, Suit, Rank
from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, str_to_bitmask, translate_hand_cards, transform

from simulated_player import TIMEOUT_SECOND
from new_simulated_player import MonteCarloPlayer7
from strategy_play import greedy_choose
from expert_play import expert_choose
from mcts import MCTS, policy_value_fn


class RiderPlayer(MonteCarloPlayer7):
    """AI player based on MCTS"""
    def __init__(self, self_player_idx, verbose=False, c_puct=2):
        super(RiderPlayer, self).__init__(verbose=verbose)

        self.mcts = MCTS(policy_value_fn, self_player_idx, c_puct)


    def reset(self):
        super(RiderPlayer, self).reset()

        self.mcts.update_with_move(-1)


    def see_played_trick(self, card):
        super(RiderPlayer, self).see_played_trick(card)

        card = tuple([SUIT_TO_INDEX[card.suit.__repr__()], NUM_TO_INDEX[card.rank.__repr__()]])

        self.mcts.update_with_move(card)


    def get_simple_game_info(self, state):
        hand_cards = [[] if player_idx != self.position else state._player_hands[player_idx] for player_idx in range(4)]

        remaining_cards = Deck().cards
        for card in state.players[0].seen_cards + hand_cards[self.position]:
            remaining_cards.remove(card)

        score_cards = []
        for player_idx, cards in enumerate(state._cards_taken):
            score_cards.append(card_to_bitmask(cards))

        init_trick = [[None, state.trick]]
        for trick_idx, (winner_index, trick) in enumerate(init_trick):
            for card_idx, card in enumerate(trick):
                for suit, rank in str_to_bitmask([card]).items():
                    trick[card_idx] = [suit, rank]

        void_info = {}
        for player_idx, info in enumerate(state.lacking_cards):
            if player_idx != self.position:
                void_info[player_idx] = info

        must_have = state.players[self.position].transfer_cards

        selection_func = np.random.choice([expert_choose, greedy_choose], size=4, p=[0.5, 0.5])

        return hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        game.are_hearts_broken()

        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            self.get_simple_game_info(copy.deepcopy(game))

        played_card = \
            self.mcts.get_move(hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func, game.trick_nr+1, game.is_heart_broken, game.expose_heart_ace, True)

        played_card_str = transform(INDEX_TO_NUM[played_card[1]], INDEX_TO_SUIT[played_card[0]])

        self.say("Hand card: {}, Validated card: {}, Picked card: {}", hand_cards, valid_cards, played_card_str)

        #self.mcts.print_tree()

        played_card = played_card_str

        return played_card
