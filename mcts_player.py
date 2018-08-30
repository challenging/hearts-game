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

from simulated_player import MonteCarloPlayer3
from player import SimplePlayer, StupidPlayer

TIMEOUT_SECOND = 0.95
COUNT_CPU = mp.cpu_count()


class MCTSPlayer(MonteCarloPlayer3):
    def __init__(self, verbose=False):
        super(MCTSPlayer, self).__init__(verbose=verbose)

        self.C = 1.4


    def play_card(self, hand_cards, game, simulation_time_limit=TIMEOUT_SECOND):
        valid_cards = self.get_valid_cards(hand_cards, game.trick, game.trick_nr, game.is_heart_broken)

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
            for card in valid_cards:
                seen_cards = [c for c in self.seen_cards[:]] + [card]
                moves_states.append((card, tuple(seen_cards)))

            average_score, played_card = sys.maxsize, None
            for card, states in moves_states:
                k = (game.current_player_idx, states)

                if k in plays:
                    avg_score = scores.get(k, 0) / plays[k]
                    if avg_score < average_score:
                        average_score = avg_score
                        played_card = card

                    self.say("{}, pick {}: score ---> {}/{}={:.2f}", valid_cards, card, scores[k], plays[k], avg_score)
                else:
                    self.say("not found {} in {}", k, plays)

            self.say("(plays, scores, played_card, valid_cards) = ({}, {}, {}({:.2f}), {})",
                len(plays), len(scores), played_card, average_score, valid_cards)
        else:
            played_card = valid_cards[0]
            self.say("don't need simulation, can only play {} card", played_card)

        return played_card


    """
    def score_func(self, scores):
        for idx, (player_idx, second_score) in enumerate(sorted(zip(range(4), scores), key=lambda x: x[1])):
            if idx == 1:
                break

        min_score, self_score = min(scores), scores[self.position]
        if self_score == min_score:
            return self_score-second_score
        else:
            return self_score-min_score
    """


    def run_simulation(self, game, plays, scores):
        hand_cards = game._player_hands[game.current_player_idx]
        remaining_cards = self.get_remaining_cards(hand_cards)

        seen_cards = copy.deepcopy(game.players[0].seen_cards)
        game.verbose = False
        game.players = [SimplePlayer() for idx in range(4)]
        for player in game.players:
            player.seen_cards = copy.deepcopy(seen_cards)

        self.redistribute_cards(game, remaining_cards[:])

        visited_states = set()

        expand  = True
        tmp_plays, tmp_scores = {}, {}

        is_first = True
        winners = game.get_game_winners()
        while not winners:
            player_idx = game.current_player_idx
            player = game.players[player_idx]

            played_card, state = None, None
            if player_idx == self.position:
                valid_cards = player.get_valid_cards(game._player_hands[game.current_player_idx], game.trick, game.trick_nr, game.is_heart_broken)

                is_all_pass, log_total = True, 0
                moves_states = []
                for card in valid_cards:
                    seen_cards = [c for c in player.seen_cards[:]] + [card]

                    key = (card, tuple(seen_cards))
                    moves_states.append(key)

                    is_all_pass &= (key in plays)
                    if is_all_pass:
                        log_total += plays[key]

                if is_all_pass:
                    log_total = np.log(log_total)

                    value, played_card, state = max(
                        ((scores[(player_idx, state)] / plays[(player_idx, state)]) + self.C * np.sqrt(log_total / plays[(player_idx, state)]),
                          card,
                          state)
                        for card, state in moves_states)
                else:
                    if is_first:
                        played_card = choice(valid_cards)

                        is_first = False
                    else:
                        played_card = player.play_card(game._player_hands[player_idx], game)

                    state = tuple(game.players[0].seen_cards[:]) + (played_card,)
            else:
                played_card = player.play_card(game._player_hands[player_idx], game)

            game.step(played_card)

            if player_idx == self.position:
                if expand and (player_idx, state) not in plays and (player_idx, state) not in tmp_plays:
                    expand = False

                    tmp_plays[(player_idx, state)] = 0
                    tmp_scores[(player_idx, state)] = 0

                visited_states.add(state)

            winners = game.get_game_winners()

        for state in visited_states:
            key = (self.position, state)
            if key in tmp_plays or key in plays:
                tmp_plays.setdefault(key, 0)
                tmp_plays[key] += 1

                tmp_scores.setdefault(key, 0)
                tmp_scores[key] += self.score_func(game.player_scores)

        return tmp_plays, tmp_scores
