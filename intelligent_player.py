"""This module containts the abstract class Player and some implementations."""
import time

import numpy as np

from card import bitmask_to_card, card_to_bitmask

from strategy_play import random_choose

from dragon_rider_player import RiderPlayer
from simulated_player import TIMEOUT_SECOND
from intelligent_mcts import IntelligentMCTS

from nn_utils import log_softmax


class IntelligentPlayer(RiderPlayer):
    def __init__(self, policy, c_puct, mcts=None, is_self_play=False, min_times=32, verbose=False):
        super(IntelligentPlayer, self).__init__(policy, c_puct, verbose=verbose)

        self.is_self_play = is_self_play
        self.min_times = min_times

        if self.is_self_play:
            self.mcts = mcts
        else:
            self.mcts = IntelligentMCTS(self.policy, self.position, self.c_puct, min_times=self.min_times)


    def reset(self, mcts=None):
        super(RiderPlayer, self).reset()

        if self.is_self_play:
            self.mcts.root_node._children = {}
            self.mcts.start_node = self.mcts.root_node
        else:
            self.mcts = IntelligentMCTS(self.policy, self.position, self.c_puct, min_times=self.min_times)


    def see_played_trick(self, card, game):
        super(RiderPlayer, self).see_played_trick(card, game)

        card = (card.suit.value, 1<<(card.rank.value-2))
        self.mcts.update_with_move(card, game.current_player_idx)


    def get_simple_game_info(self, state):
        hand_cards, trick_cards, init_trick, must_have, _ = super(IntelligentPlayer, self).get_simple_game_info(state)

        return hand_cards, trick_cards, init_trick, must_have, [random_choose]*4


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        stime = time.time()

        if game.trick_nr == 0 and len(game.trick) == 0:
            for card in game._player_hands[self.position]:
                if card in self.remaining_cards:
                    self.remaining_cards.remove(card)

        game.are_hearts_broken()

        hand_cards, trick_cards, init_trick, must_have, selection_func = \
            self.get_simple_game_info(game)

        vcards = self.get_valid_cards(game._player_hands[self.position], game)
        bit_vcards = card_to_bitmask(vcards)

        etime = simulation_time_limit
        if self.is_self_play:
            etime = len(vcards)*simulation_time_limit
        else:
            etime = 4

        self.mcts._self_player_idx = self.position
        print(self.mcts)

        results = \
            self.mcts.get_move(game.current_player_idx, 
                               hand_cards, 
                               vcards, 
                               self.remaining_cards, 
                               game._b_cards_taken,
                               trick_cards,
                               self.num_hand_cards, 
                               init_trick, 
                               self.void_info, 
                               must_have, 
                               selection_func, 
                               game.trick_nr+1, 
                               game.is_heart_broken, 
                               [2 if player.expose else 1 for player in game.players], 
                               False, 
                               etime, 
                               True,
                               False)

        valid_cards, valid_probs = [], []
        for card, info in sorted(results.items(), key=lambda x: x[1]):
            suit, rank = card
            n_visits, value = info[0], info[1]
            if n_visits > 0 and bit_vcards[suit] & rank:
                valid_cards.append(bitmask_to_card(card[0], card[1]))
                valid_probs.append(n_visits)

                if not self.is_self_play and valid_probs[-1] > 0:
                    self.say("Player-{}, played card: {}, {} times", self.position, valid_cards[-1], valid_probs[-1])

        if not valid_probs:
            print(results)

        valid_probs = log_softmax(valid_probs)

        if self.is_self_play:
            cards, probs = [], []
            for card, info in sorted(results.items(), key=lambda x: x[1]):
                n_visits, value = info[0], info[1]

                cards.append(bitmask_to_card(card[0], card[1]))
                probs.append(n_visits)

                if probs[-1] > 0:
                    self.say("Player-{}, played card: {}, {} times, value={}", self.position, cards[-1], probs[-1], value)

            #probs = softmax(probs, temp)

            move = np.random.choice(
                    valid_cards,
                    p=0.75*valid_probs + 0.25*np.random.dirichlet(0.3*np.ones(len(valid_probs))))

            self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", \
                time.time()-stime, hand_cards, valid_cards, move)

            return move, zip(cards, probs)
        else:
            move, scenario = None, None
            if valid_cards:
                move = valid_cards[np.argmax(valid_probs)]
                scenario = "1"
            else:
                move = np.random.choice(vcards)
                scenario = "2"

            self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}, Scenario: {}", \
                time.time()-stime, hand_cards, valid_cards, move, scenario)

            self.mcts.update_with_move(-1, game.current_player_idx)

            return move
