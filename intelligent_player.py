"""This module containts the abstract class Player and some implementations."""
import time

import numpy as np

from card import INDEX_TO_NUM, INDEX_TO_SUIT
from card import transform

from dragon_rider_player import RiderPlayer
from simulated_player import TIMEOUT_SECOND


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)

    return probs


class IntelligentPlayer(RiderPlayer):
    """AI player based on MCTS"""
    def __init__(self, policy, c_puct, verbose=False):
        super(RiderPlayer, self).__init__(verbose=verbose)

        self.policy = policy
        self.c_puct = c_puct


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND, temp=1e-3):
        stime = time.time()

        game.are_hearts_broken()

        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            self.get_simple_game_info(game)

        valid_cards = self.get_valid_cards(game._player_hands[self.position], game)

        results = \
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
                               False, 
                               simulation_time_limit,
                               True)

        cards, probs = [], []
        for played_card, n_visits in sorted(results.items(), key=lambda x: x[1]):
            cards.append(transform(INDEX_TO_NUM[played_card[1]], INDEX_TO_SUIT[played_card[0]]))
            probs.append(n_visits)

        probs = softmax(1/temp*np.log(np.array(probs) + 1e-10))


        self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", \
            time.time()-stime, hand_cards, valid_cards, played_card)

        return transform(INDEX_TO_NUM[played_card[1]], INDEX_TO_SUIT[played_card[0]]), zip(cards, probs)
