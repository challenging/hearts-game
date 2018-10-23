"""This module containts the abstract class Player and some implementations."""
import sys

import time

#import numpy as np
from random import choice, shuffle

from collections import defaultdict

from card import Deck, Card, Suit, Rank
from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, str_to_bitmask, translate_hand_cards, transform

from simulated_player import TIMEOUT_SECOND
from new_simulated_player import MonteCarloPlayer7
from strategy_play import greedy_choose, random_choose
from expert_play import expert_choose
from mcts import MCTS
from mcts import policy_value_fn


class RiderPlayer(MonteCarloPlayer7):
    """AI player based on MCTS"""
    def __init__(self, verbose=False, c_puct=2):
        super(RiderPlayer, self).__init__(verbose=verbose)

        self.c_puct = c_puct


    def set_position(self, idx):
        super(RiderPlayer, self).set_position(idx)

        self.mcts = MCTS(policy_value_fn, self.position, self.c_puct)


    def reset(self):
        super(RiderPlayer, self).reset()

        self.mcts.update_with_move(-1)


    def see_played_trick(self, card, game):
        super(RiderPlayer, self).see_played_trick(card, game)

        card = tuple([SUIT_TO_INDEX[card.suit.__repr__()], NUM_TO_INDEX[card.rank.__repr__()]])

        self.mcts.update_with_move(card)

        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            self.get_simple_game_info(game)

        if remaining_cards and len(init_trick[-1][1]) < 4 and len(game._player_hands[self.position]) > 0 and len(game._player_hands[self.position]) < 13:
            try:
                self.mcts.get_move(hand_cards, 
                                   remaining_cards, 
                                   score_cards, 
                                   init_trick, 
                                   void_info, 
                                   must_have, 
                                   selection_func, 
                                   game.trick_nr+1, 
                                   game.is_heart_broken, 
                                   game.expose_heart_ace, 
                                   True, 
                                   0.15,
                                   False)
            except:
                self.mcts = MCTS(policy_value_fn, self.position, self.c_puct)

                print("error in seen_cards")


    def get_simple_game_info(self, state):
        hand_cards = [[] if player_idx != self.position else state._player_hands[player_idx] for player_idx in range(4)]

        remaining_cards = Deck().cards
        for card in state.players[0].seen_cards + hand_cards[self.position]:
            remaining_cards.remove(card)

        score_cards = []
        for player_idx, cards in enumerate(state._cards_taken):
            score_cards.append(card_to_bitmask(cards))

        init_trick = [[None, state.trick[:]]]
        for trick_idx, (winner_index, trick) in enumerate(init_trick):
            for card_idx, card in enumerate(trick):
                for suit, rank in str_to_bitmask([card]).items():
                    trick[card_idx] = [suit, rank]

        void_info = {}
        for player_idx, info in enumerate(state.lacking_cards):
            if player_idx != self.position:
                void_info[player_idx] = info

        must_have = state.players[self.position].transfer_cards

        #selection_func = [choice([expert_choose, greedy_choose]) for _ in range(4)]

        num_of_suit = [0, 0, 0, 0]
        for card in state._player_hands[self.position]:
            num_of_suit[card.suit.value] += 1

        if state.trick_nr < 5 and any([True if num < 2 or num > 5 else False for num in num_of_suit]):
            selection_func = [greedy_choose]*4
        else:
            selection_func = [choice([expert_choose, greedy_choose]) for _ in range(4)]
            shuffle(selection_func)
        #print(selection_func)

        return hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        stime = time.time()

        game.are_hearts_broken()

        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            self.get_simple_game_info(game)

        played_card = \
            self.mcts.get_move(hand_cards, 
                               remaining_cards, 
                               score_cards, 
                               init_trick, 
                               void_info, 
                               must_have, 
                               selection_func, 
                               game.trick_nr+1, 
                               game.is_heart_broken, 
                               game.expose_heart_ace, 
                               True, 
                               simulation_time_limit,
                               True)

        played_card = transform(INDEX_TO_NUM[played_card[1]], INDEX_TO_SUIT[played_card[0]])

        self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", \
            time.time()-stime, hand_cards, valid_cards, played_card)

        #self.mcts.print_tree()

        return played_card
