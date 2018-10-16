"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import numpy as np

from card import INDEX_TO_NUM, INDEX_TO_SUIT, SUIT_TO_INDEX, NUM_TO_INDEX
from card import transform, translate_hand_cards

from mcts import MCTS, policy_value_fn
from simulated_player import TIMEOUT_SECOND
from new_simulated_player import MonteCarloPlayer7


class IntelligentPlayer(MonteCarloPlayer7):
    """AI player based on MCTS"""
    def __init__(self, self_player_idx, verbose=False, c_puct=2):
        super(IntelligentPlayer, self).__init__(verbose=verbose)

        self.mcts = MCTS(policy_value_fn, self_player_idx, c_puct)


    def reset(self):
        super(IntelligentPlayer, self).reset()

        self.mcts.update_with_move(-1)


    def see_played_trick(self, card):
        super(IntelligentPlayer, self).see_played_trick(card)

        card = tuple([SUIT_TO_INDEX[card.suit.__repr__()], NUM_TO_INDEX[card.rank.__repr__()]])

        self.mcts.update_with_move(card)


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        game.are_hearts_broken()

        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        results = self.mcts.get_move(copy.deepcopy(game))

        for played_card, info in sorted(results.items(), key=lambda x: -x[1][0]):
            break

        played_card_str = transform(INDEX_TO_NUM[played_card[1]], INDEX_TO_SUIT[played_card[0]])

        self.say("Hand card: {}, Validated card: {}, Picked card: {}", hand_cards, valid_cards, played_card_str)

        self.mcts.print_tree()

        played_card = played_card_str

        return played_card, results
