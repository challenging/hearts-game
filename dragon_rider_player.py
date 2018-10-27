"""This module containts the abstract class Player and some implementations."""
import sys

import time

#import numpy as np
from random import choice, shuffle

from collections import defaultdict

from card import Deck, Card, Suit, Rank
from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, bitmask_to_card, str_to_bitmask

from simulated_player import TIMEOUT_SECOND
from new_simulated_player import MonteCarloPlayer7
from strategy_play import greedy_choose, random_choose
from expert_play import expert_choose
from mcts import MCTS


class RiderPlayer(MonteCarloPlayer7):
    """AI player based on MCTS"""
    def __init__(self, policy, c_puct, verbose=False):
        super(RiderPlayer, self).__init__(verbose=verbose)

        self.policy = policy
        self.c_puct = c_puct


    def set_position(self, idx):
        super(RiderPlayer, self).set_position(idx)

        #if not hasattr(self, "mcts"):
        #    self.mcts = MCTS(policy, self.position, self.c_puct)

        self.mcts = MCTS(self.policy, self.position, self.c_puct)


    def reset(self):
        super(RiderPlayer, self).reset()

        #self.mcts.start_node = self.mcts._root
        self.mcts = MCTS(self.policy, self.position, self.c_puct)


    def see_played_trick(self, card, game):
        super(RiderPlayer, self).see_played_trick(card, game)

        #self.say("steal time({}) to simulate to game results", card)

        card = tuple([SUIT_TO_INDEX[card.suit.__repr__()], NUM_TO_INDEX[card.rank.__repr__()]])

        self.mcts.update_with_move(card)

        valid_cards = self.get_valid_cards(game._player_hands[self.position], game)
        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            self.get_simple_game_info(game)

        if len(game._player_hands[self.position]) > 0:
            try:
                self.mcts.get_move(hand_cards, 
                                   valid_cards,
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
                                   0.185,
                                   False)
            except Exception as e:
                #self.mcts = MCTS(policy, self.position, self.c_puct)
                #self.mcts.start_node = self.mcts._root

                self.say("error in seen_cards: {}", e)

                #raise
        #else:
        #    if game.get_game_winners():
        #        rating = get_rating(game.player_scores)

        #        self.mcts.start_node.update_recursive(rating)


    def get_simple_game_info(self, state):
        hand_cards = [[] if player_idx != self.position else state._player_hands[player_idx] for player_idx in range(4)]

        remaining_cards = Deck().cards
        for card in state.players[self.position].seen_cards + hand_cards[self.position]:
            if card in remaining_cards: remaining_cards.remove(card)

        score_cards = []
        for player_idx, cards in enumerate(state._cards_taken):
            score_cards.append(card_to_bitmask(cards))

        init_trick = [[None, state.trick[:]]]
        for trick_idx, (winner_index, trick) in enumerate(init_trick):
            for card_idx, card in enumerate(trick):
                for suit, rank in str_to_bitmask([card]).items():
                    trick[card_idx] = [suit, rank]

        is_void, void_info = False, {}
        for player_idx, info in enumerate(state.lacking_cards):
            if player_idx != self.position:
                void_info[player_idx] = info

                is_void |= any([v for v in info.values()])

        must_have = state.players[self.position].transfer_cards

        selection_func = [random_choose]

        return hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        stime = time.time()

        self.say("Player-{}, the information of lacking_card is {}", \
            self.position, [(player_idx, k) for player_idx, info in enumerate(game.lacking_cards) for k, v in info.items() if v])

        game.are_hearts_broken()

        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            self.get_simple_game_info(game)

        valid_cards = self.get_valid_cards(game._player_hands[self.position], game)

        played_card = \
            self.mcts.get_move(hand_cards, 
                               valid_cards,
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

        played_card = bitmask_to_card(played_card[0], played_card[1])

        self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", \
            time.time()-stime, hand_cards, valid_cards, played_card)

        return played_card
