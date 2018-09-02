"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np
import multiprocessing as mp

from collections import defaultdict
from random import shuffle, choice

from card import Suit, Rank, Card, Deck
from rules import is_card_valid

from simulated_player import MonteCarloPlayer4
from player import StupidPlayer, SimplePlayer

TIMEOUT_SECOND = 0.9
COUNT_CPU = mp.cpu_count()


class MCTSPlayer(MonteCarloPlayer4):
    def __init__(self, verbose=False):
        super(MCTSPlayer, self).__init__(verbose=verbose)

        self.C = 1.5


    def play_card(self, hand_cards, game, simulation_time_limit=TIMEOUT_SECOND):
        game.are_hearts_broken()
        valid_cards = self.get_valid_cards(hand_cards, game)

        card = None
        if len(valid_cards) > 1:
            stime = time.time()

            plays, scores = defaultdict(int), defaultdict(int)
            pool = mp.Pool(processes=self.num_of_cpu)
            while time.time() - stime < simulation_time_limit:
                mul_result = [pool.apply_async(self.run_simulation, args=(game, plays, scores)) for _ in range(self.num_of_cpu)]
                results = [res.get() for res in mul_result]

                for tmp_plays, tmp_scores in results:
                    for k, v in tmp_plays.items():
                        plays[k] += v

                    for k, v in tmp_scores.items():
                        scores[k] += v

            pool.close()

            moves_states = []
            for card in sorted(valid_cards, key=lambda x: -(x.suit.value*13+x.rank.value)):
                seen_cards = [c for c in self.seen_cards[:]] + [card]
                moves_states.append((card, tuple(seen_cards)))

            average_score, played_card = sys.maxsize, None
            for state in moves_states:
                if state in plays:
                    avg_score = scores.get(state, 0) / plays[state]
                    if avg_score < average_score:
                        average_score = avg_score
                        played_card = state[0]

                    self.say("{}, pick {}: score ---> {}/{}={:.2f}", valid_cards, state[0], scores[state], plays[state], avg_score)
                else:
                    self.say("not found {} in {}", state, plays)

            self.say("(plays, scores, played_card, valid_cards) = ({}, {}, {}({:.2f}), {})",
                len(plays), len(scores), played_card, average_score, valid_cards)
        else:
            played_card = valid_cards[0]
            self.say("don't need simulation, can only play {} card", played_card)

        return played_card


    def run_simulation(self, game, plays, scores):
        hand_cards = game._player_hands[game.current_player_idx]
        remaining_cards = self.get_remaining_cards(hand_cards)

        seen_cards = copy.deepcopy(game.players[0].seen_cards)
        game.verbose = False
        game.players = [StupidPlayer() for idx in range(4)]
        for player in game.players:
            player.seen_cards = copy.deepcopy(seen_cards)

        trick_nr = game.trick_nr

        self.redistribute_cards(game, remaining_cards[:])

        tmp_plays, tmp_scores = {}, {}

        player = game.players[self.position]

        played_card = None
        valid_cards = player.get_valid_cards(game._player_hands[self.position], game)

        is_all_pass, log_total = True, 0
        moves_states = []
        for card in valid_cards:
            tmp_seen_cards = [c for c in seen_cards[:]] + [card]

            key = (card, tuple(tmp_seen_cards))
            moves_states.append(key)

            is_all_pass &= (key in plays)
            if is_all_pass:
                log_total += plays[key]

        if is_all_pass:
            log_total = np.log(log_total)

            value, state = max(
                (-(scores[state] / plays[state]) + self.C * np.sqrt(log_total / plays[state]), state) for state in moves_states)

            played_card = state[0]
        else:
            played_card = player.play_card(game._player_hands[self.position], game)

        state = (played_card, tuple(game.players[0].seen_cards[:]) + (played_card,))

        game.step(played_card)

        for _ in range(4-len(game.trick)):
            game.step()

        for _ in range(13-game.trick_nr):
            game.play_trick()

        if trick_nr < 6:
            self.overwrite_game_score_func(game)

        game.score()

        tmp_plays[state] = 1
        tmp_scores[state] = self.score_func(game.player_scores)

        return tmp_plays, tmp_scores
