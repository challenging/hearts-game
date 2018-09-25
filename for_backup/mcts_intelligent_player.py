"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np
import multiprocessing as mp

from scipy.stats import describe
from collections import defaultdict
from random import shuffle, choice

from game import Game
from card import Suit, Rank, Card, Deck
from rules import is_card_valid

from simulated_player import MonteCarloPlayer6
from player import StupidPlayer, SimplePlayer

from nn_utils import card2v, v2card


TIMEOUT_SECOND = 0.93
COUNT_CPU = mp.cpu_count()


class IntelligentPlayer(MonteCarloPlayer6):
    def __init__(self, C, policy, verbose=False):
        super(IntelligentPlayer, self).__init__(verbose=verbose)

        self.C = C

        self.policy = policy


    def play_card(self, game, simulation_time_limit=TIMEOUT_SECOND):
        game.are_hearts_broken()

        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        card = None
        if len(valid_cards) > 1:
            stime = time.time()

            local_game = Game([SimplePlayer(verbose=False) for idx in range(4)], verbose=False)
            for player in local_game.players:
                player.seen_cards = copy.deepcopy(self.seen_cards)
            local_game.trick = game.trick[:]
            local_game.trick_nr = game.trick_nr
            local_game.current_player_idx = game.current_player_idx

            local_game.take_pig_card = game.take_pig_card
            local_game.is_heart_broken = game.is_heart_broken
            local_game.is_shootmoon = game.is_shootmoon

            local_game._player_hands = game._player_hands[:]
            local_game._cards_taken = game._cards_taken[:]

            scores, scores_num = defaultdict(float), defaultdict(int)
            #pool = mp.Pool(processes=self.num_of_cpu)
            while time.time() - stime < simulation_time_limit:
                for k, v in self.run_simulation(copy.deepcopy(local_game), scores, scores_num).items():
                    scores_num[k] += 1
                    scores[k] += (v-scores[k])/scores_num[k]

                #mul_result = [pool.apply_async(self.run_simulation, args=(game, scores)) for _ in range(self.num_of_cpu)]
                #results = [res.get() for res in mul_result]

                #for tmp_scores in results:
                #    for k, v in tmp_scores.items():
                #        scores[k].append(v)
            #pool.close()

            moves_states = []
            for card in valid_cards:
                seen_cards = [c for c in self.seen_cards[:]] + [card]
                moves_states.append((card, tuple(seen_cards)))

            average_score, played_card = -sys.maxsize, None
            for state in moves_states:
                if state in scores:
                    avg_score = scores[state]

                    if avg_score > average_score:
                        average_score = avg_score
                        played_card = state[0]

                    self.say("{}, pick {}: (n={}, Q={:.4f})", valid_cards, state[0], scores_num[state], scores[state])
                else:
                    self.say("not found {} in {}", state, move_states)
        else:
            played_card = self.no_choice(valid_cards[0])

        return played_card


    def run_simulation(self, game, scores, scores_num):
        hand_cards = game._player_hands[game.current_player_idx]
        remaining_cards = self.get_remaining_cards(hand_cards)

        seen_cards = copy.deepcopy(game.players[0].seen_cards)
        game.verbose = False
        game.players = [StupidPlayer() for idx in range(4)]
        for player in game.players:
            player.seen_cards = copy.deepcopy(seen_cards)

        trick_nr = game.trick_nr

        game = self.redistribute_cards(game, remaining_cards[:])

        tmp_plays, tmp_scores = {}, defaultdict(float)

        player = game.players[self.position]

        played_card = None

        valid_cards = player.get_valid_cards(game._player_hands[self.position], game)
        cards = np.zeros(13, dtype=np.int32)
        for idx, card in enumerate(valid_cards):
            cards[idx] = card2v(card)

        action_probs, rating = self.policy([game.current_status()], [cards])
        action_probs, rating = action_probs[0], rating[0]

        #for card, prob in zip(valid_cards, action_probs):
        #    print(card, prob)

        is_all_pass, total = True, 0
        moves_states = []
        for card in valid_cards:
            tmp_seen_cards = [c for c in seen_cards[:]] + [card]

            key = (card, tuple(tmp_seen_cards))
            moves_states.append(key)

            is_all_pass &= (key in scores)
            if is_all_pass:
                total += scores_num[key]

        if is_all_pass:
            value, state = max(
                (scores[state] + self.C * action_probs[idx] * np.sqrt(total) / (scores_num[state]+1), state) for idx, state in enumerate(moves_states))

            played_card = state[0]
        else:
            played_card = choice(valid_cards)

        state = (played_card, tuple(game.players[0].seen_cards[:]) + (played_card,))

        game.step(played_card)

        tmp_scores[state] = rating[self.position]

        return tmp_scores
