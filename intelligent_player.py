"""This module containts the abstract class Player and some implementations."""
import time

import numpy as np

from card import INDEX_TO_NUM, INDEX_TO_SUIT
from card import bitmask_to_card

from dragon_rider_player import RiderPlayer
from simulated_player import TIMEOUT_SECOND
from intelligent_mcts import IntelligentMCTS


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)

    return probs


class IntelligentPlayer(RiderPlayer):
    """AI player based on MCTS"""
    def __init__(self, policy, c_puct, is_self_play=False, verbose=False):
        super(IntelligentPlayer, self).__init__(policy, c_puct, verbose=verbose)

        self.is_self_play = is_self_play


    def set_position(self, idx):
        super(RiderPlayer, self).set_position(idx)

        self.mcts = IntelligentMCTS(self.policy, self.position, self.c_puct)


    def reset(self):
        super(RiderPlayer, self).reset()

        self.mcts = IntelligentMCTS(self.policy, self.position, self.c_puct)


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

        #print("results", results)
        #self.mcts.print_tree()

        valid_cards, valid_probs = [], []
        for card, n_visits in sorted(results.items(), key=lambda x: x[1]):
            if n_visits > 0:
                valid_cards.append(bitmask_to_card(card[0], card[1]))
                valid_probs.append(n_visits)
        valid_probs = softmax(1/temp*np.log(np.array(valid_probs) + 1e-10))

        if self.is_self_play:
            cards, probs = [], []
            for card, n_visits in sorted(results.items(), key=lambda x: x[1]):
                cards.append(bitmask_to_card(card[0], card[1]))
                probs.append(n_visits)

            probs = softmax(1/temp*np.log(np.array(probs) + 1e-10))

            move = np.random.choice(
                    valid_cards,
                    p=0.75*valid_probs + 0.25*np.random.dirichlet(0.3*np.ones(len(valid_probs))))

            return move, zip(cards, probs)
        else:
            move = np.random.choice(valid_cards, p=valid_probs)

            self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", \
                time.time()-stime, hand_cards, valid_cards, move)

            return move
