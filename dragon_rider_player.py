"""This module containts the abstract class Player and some implementations."""
import os
import sys

import time
import glob
import pickle

from random import choice, shuffle

from collections import defaultdict

from card import Deck, Card, Suit, Rank, HEARTS_A, HEARTS_K, HEARTS_Q
from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, bitmask_to_card, str_to_bitmask

from rules import get_rating

from simulated_player import TIMEOUT_SECOND
from new_simulated_player import MonteCarloPlayer7
from strategy_play import greedy_choose, random_choose
from expert_play import expert_choose

from mcts import MCTS


BASEPATH = "memory_tree"
BASEPATH_MODEL = os.path.join(BASEPATH, "model")

if not os.path.exists(BASEPATH_MODEL):
    os.makedirs(BASEPATH_MODEL)


class RiderPlayer(MonteCarloPlayer7):
    def __init__(self, policy, c_puct, verbose=False, freq_save=64):
        super(RiderPlayer, self).__init__(verbose=verbose)

        self.policy = policy
        self.c_puct = c_puct

        self.freq_idx = 1
        self.freq_save = freq_save

        self.mcts = None


    def reset(self):
        super(RiderPlayer, self).reset()

        global BASEPATH_MODEL

        if self.mcts is None:
            self.mcts = MCTS(self.policy, self.position, self.c_puct)

            for filepath_in in sorted(glob.glob("{}/*pkl".format(BASEPATH_MODEL)), key=os.path.getmtime)[::-1]:
                self.say("load memory from {}", filepath_in)
                with open(filepath_in, "rb") as in_file:
                    self.mcts.root_node = pickle.load(in_file)
                    self.mcts.start_node = self.mcts.root_node

                break

        else:
            self.mcts.start_node = self.mcts.root_node

            self.freq_idx += 1


    def expose_hearts_ace(self, hand_cards):
        if HEARTS_A in hand_cards:
            points, safe_hearts, hearts = 0, 0, []
            for card in hand_cards:
                if card.suit == Suit.hearts:
                    if card.rank > Rank.nine:
                        safe_hearts -= 1
                    else:
                        safe_hearts += 1

                    hearts.append(card)

                points += max(0, card.rank.value-10)

            if points < 12 and safe_hearts > 0:
                self.say("scenario - 1, points: {}, safe_hearts: {}", points, safe_hearts)

                self.expose = True
            elif points > 21:
                if len(hearts) > 0:
                    self.say("scenario - 2.1, points: {}, hearts: {}", points, hearts)

                    self.expose = True
                elif len(hearts) > 1 and HEARTS_K in hearts:
                    self.say("scenario - 2.2, points: {}, hearts: {}", points, hearts)

                    self.expose = True
                elif len(hearts) > 2 and all([card in hearts for card in [HEARTS_K, HEARTS_Q]]):
                    self.say("scenario - 2.3, points: {}, hearts: {}", points, hearts)
                    self.expose = True

        return self.expose


    def see_played_trick(self, card, game):
        super(RiderPlayer, self).see_played_trick(card, game)

        card = (card.suit.value, 1<<(card.rank.value-2))
        self.mcts.update_with_move(card, game.current_player_idx)

        valid_cards = self.get_valid_cards(game._player_hands[self.position], game)
        hand_cards, historical_cards, init_trick, must_have, selection_func = \
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

        #self.say("steal time({}, {:.2} seconds) to simulate to game results, {}, {}, {}", \
        #    card, steal_time, first_player_idx, len(self.remaining_cards), self.num_hand_cards)

        if len(game._player_hands[self.position]) > 0:
            self.mcts.get_move(first_player_idx, 
                               hand_cards, 
                               valid_cards,
                               self.remaining_cards, 
                               game._b_cards_taken, 
                               historical_cards,
                               self.num_hand_cards, 
                               init_trick, 
                               self.void_info, 
                               must_have, 
                               selection_func, 
                               game.trick_nr+1, 
                               game.is_heart_broken, 
                               [2 if player.expose else 1 for player in game.players], 
                               True, 
                               steal_time,
                               False,
                               False)
        else:
            if game.get_game_winners():
                self.mcts.start_node.update_recursive(get_rating(game.player_scores))


    def get_simple_game_info(self, state):
        hand_cards = [[] if player_idx != self.position else state._player_hands[player_idx] for player_idx in range(4)]

        trick_cards = state.trick_cards

        init_trick = [[None, state.trick[:]]]
        for trick_idx, (winner_index, trick) in enumerate(init_trick):
            for card_idx, card in enumerate(trick):
                for suit, rank in str_to_bitmask([card]).items():
                    trick[card_idx] = [suit, rank]

        must_have = state.players[self.position].transfer_cards

        selection_func = [expert_choose]*4

        return hand_cards, trick_cards, init_trick, must_have, selection_func


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        stime = time.time()

        if game.trick_nr == 0 and len(game.trick) == 0:
            for card in game._player_hands[self.position]:
                if card in self.remaining_cards:
                    self.remaining_cards.remove(card)

        self.say("Player-{}, the information of lacking_card is {}", \
            self.position, [(player_idx, k) for player_idx, info in enumerate(game.lacking_cards) for k, v in info.items() if v])

        game.are_hearts_broken()

        hand_cards, trick_cards, init_trick, must_have, selection_func = \
            self.get_simple_game_info(game)

        valid_cards = self.get_valid_cards(game._player_hands[self.position], game)

        played_card = \
            self.mcts.get_move(game.current_player_idx,
                               hand_cards, 
                               valid_cards,
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
                               True, 
                               simulation_time_limit,
                               True,
                               False)

        played_card = bitmask_to_card(played_card[0], played_card[1])

        """
        from render_tree import get_tree 
        tree = get_tree(self.mcts.start_node)
        tree.show()
        print("depth={}".format(tree.depth()))
        """

        self.say("Cost: {:.4f} seconds, Hand card: {}, Validated card: {}, Picked card: {}", \
            time.time()-stime, hand_cards, valid_cards, played_card)

        return played_card
