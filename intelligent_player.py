"""This module containts the abstract class Player and some implementations."""
import time

import numpy as np

from card import INDEX_TO_NUM, INDEX_TO_SUIT
from card import bitmask_to_card, card_to_bitmask

from strategy_play import random_choose

from dragon_rider_player import RiderPlayer
from simulated_player import TIMEOUT_SECOND
from new_simulated_player import MonteCarloPlayer7
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


    def reset(self):
        super(RiderPlayer, self).reset()

        self.mcts = IntelligentMCTS(self.policy, self.position, self.c_puct)


    def get_simple_game_info(self, game):
        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            super(IntelligentPlayer, self).get_simple_game_info(game)

        selection_func = [random_choose]

        return hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND, temp=1e-3):
        stime = time.time()

        game.are_hearts_broken()

        hand_cards, remaining_cards, score_cards, init_trick, void_info, must_have, selection_func = \
            self.get_simple_game_info(game)

        vcards = self.get_valid_cards(game._player_hands[self.position], game)
        bit_vcards = card_to_bitmask(vcards)

        results = \
            self.mcts.get_move(game.current_player_idx, 
                               hand_cards, 
                               vcards, 
                               remaining_cards, 
                               score_cards, 
                               self.num_hand_cards, 
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

        valid_cards, valid_probs = [], []
        for card, n_visits in sorted(results.items(), key=lambda x: x[1]):
            suit, rank = card
            if n_visits > 0 and bit_vcards[suit] & rank:
                valid_cards.append(bitmask_to_card(card[0], card[1]))
                valid_probs.append(n_visits)

        if valid_cards:
            valid_probs = softmax(1/temp*np.log(np.array(valid_probs) + 1e-10))
        else:
            self.say("use simple valid_cards - {}", vcards)

            valid_cards = vcards
            valid_probs = [1.0/len(valid_cards) for _ in range(len(valid_cards))]
            valid_probs = softmax(1/temp*np.log(np.array(valid_probs) + 1e-10))


        if self.is_self_play:
            cards, probs = [], []
            for card, n_visits in sorted(results.items(), key=lambda x: x[1]):
                cards.append(bitmask_to_card(card[0], card[1]))
                probs.append(n_visits)

                if probs[-1] > 0:
                    self.say("Played Card: {}, {} times", cards[-1], probs[-1])
            probs = softmax(1/temp*np.log(np.array(probs) + 1e-10))

            move = np.random.choice(
                    valid_cards,
                    p=0.75*valid_probs + 0.25*np.random.dirichlet(0.3*np.ones(len(valid_probs))))

            #print("---> played_cards", self.position, valid_cards)

            return move, zip(cards, probs)
        else:
            move = np.random.choice(valid_cards, p=valid_probs)

            self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", \
                time.time()-stime, hand_cards, valid_cards, move)

            self.mcts.update_with_move(-1)

            return move
