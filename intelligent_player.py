":""This module containts the abstract class Player and some implementations."""
import time

import numpy as np

from card import NUM_TO_INDEX, SUIT_TO_INDEX
from card import bitmask_to_card, card_to_bitmask

from strategy_play import random_choose, greedy_choose
from expert_play import expert_choose

from dragon_rider_player import RiderPlayer
from simulated_player import TIMEOUT_SECOND
from new_simulated_player import MonteCarloPlayer7
from intelligent_mcts import IntelligentMCTS


def softmax(x, temp):
    probs = x / np.sum(x)

    return probs


class IntelligentPlayer(RiderPlayer):
    """AI player based on MCTS"""
    def __init__(self, policy, c_puct, is_self_play=False, verbose=False):
        super(IntelligentPlayer, self).__init__(policy, c_puct, verbose=verbose)

        self.is_self_play = is_self_play


    def reset(self):
        super(RiderPlayer, self).reset()

        self.mcts = IntelligentMCTS(self.policy, self.position, self.c_puct)


    def see_played_trick(self, card, game):
        super(RiderPlayer, self).see_played_trick(card, game)

        card = tuple([SUIT_TO_INDEX[card.suit.__repr__()], NUM_TO_INDEX[card.rank.__repr__()]])

        self.mcts.update_with_move(card)

        valid_cards = self.get_valid_cards(game._player_hands[self.position], game)
        hand_cards, init_trick, must_have, selection_func = \
            self.get_simple_game_info(game)

        steal_time = 0.28

        first_player_idx = (game.current_player_idx+1)%4
        if len(game.trick) == 4:
            winning_index, _ = game.winning_index()
            first_player_idx = (game.current_player_idx + winning_index) % 4

            if first_player_idx == self.position:
                steal_time = 0.85
            else:
                steal_time = 0.55

        self.say("steal time({}, {:.2} seconds) to simulate to game results, {}, {}, {}", \
            card, steal_time, first_player_idx, len(self.remaining_cards), self.num_hand_cards)

        if len(game._player_hands[self.position]) > 0:
            self.mcts.get_move(first_player_idx, 
                               hand_cards, 
                               valid_cards,
                               self.remaining_cards, 
                               game._b_cards_taken, 
                               self.num_hand_cards, 
                               init_trick, 
                               self.void_info, 
                               must_have, 
                               selection_func, 
                               game.trick_nr+1, 
                               game.is_heart_broken, 
                               game.expose_heart_ace, 
                               True, 
                               steal_time,
                               False,
                               True)
        else:
            if game.get_game_winners():
                rating = get_rating(self.position, game.player_scores, game.is_shootmoon)

                self.mcts.start_node.update_recursive(rating)


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND, temp=1):
        stime = time.time()

        game.are_hearts_broken()

        hand_cards, init_trick, must_have, selection_func = \
            self.get_simple_game_info(game)

        vcards = self.get_valid_cards(game._player_hands[self.position], game)
        bit_vcards = card_to_bitmask(vcards)

        etime = simulation_time_limit
        if game.trick_nr > 11:
            etime /= 4
        elif game.trick_nr > 8:
            etime /= 3
        elif game.trick_nr > 5:
            etime /= 1.5

        results = \
            self.mcts.get_move(game.current_player_idx, 
                               hand_cards, 
                               vcards, 
                               self.remaining_cards, 
                               game._b_cards_taken,
                               self.num_hand_cards, 
                               init_trick, 
                               self.void_info, 
                               must_have, 
                               selection_func, 
                               game.trick_nr+1, 
                               game.is_heart_broken, 
                               game.expose_heart_ace, 
                               False, 
                               etime, 
                               True,
                               True)

        valid_cards, valid_probs = [], []
        for card, n_visits in sorted(results.items(), key=lambda x: x[1]):
            suit, rank = card
            if n_visits > 0 and bit_vcards[suit] & rank:
                valid_cards.append(bitmask_to_card(card[0], card[1]))
                valid_probs.append(n_visits)

                #self.say("Player-{}, played card: {}, {} times", self.position, valid_cards[-1], valid_probs[-1])

        valid_probs = softmax(valid_probs, temp)

        if self.is_self_play:
            cards, probs = [], []
            for card, n_visits in sorted(results.items(), key=lambda x: x[1]):
                cards.append(bitmask_to_card(card[0], card[1]))
                probs.append(n_visits)

                if probs[-1] > 0:
                    self.say("Player-{}, played card: {}, {} times", self.position, cards[-1], probs[-1])

            probs = softmax(probs, temp)

            move = np.random.choice(
                    valid_cards,
                    p=0.75*valid_probs + 0.25*np.random.dirichlet(0.3*np.ones(len(valid_probs))))

            self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", \
                time.time()-stime, hand_cards, valid_cards, move)

            return move, zip(cards, probs)
        else:
            move = np.random.choice(valid_cards, p=valid_probs)

            self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", \
                time.time()-stime, hand_cards, valid_cards, move)

            self.mcts.update_with_move(-1)

            return move
