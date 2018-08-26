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

from simulated_player import MonteCarloPlayer2
from player import SimplePlayer, StupidPlayer

TIMEOUT_SECOND = 0.9
COUNT_CPU = 1#mp.cpu_count()


class MCTSPlayer(MonteCarloPlayer2):
    def __init__(self, verbose=False):
        super(MCTSPlayer, self).__init__(verbose=verbose)

        self.C = 1.4
        self.max_moves = 128


    def play_card(self, hand_cards, game, simulation_time_limit=TIMEOUT_SECOND):
        valid_cards = self.get_valid_cards(hand_cards, game.trick, game.trick_nr, game.is_heart_broken)

        card = None
        if len(valid_cards) > 1:
            stime = time.time()

            count_simulation = 0
            wins, plays, scores = defaultdict(int), defaultdict(int), defaultdict(int)
            pool = mp.Pool(processes=self.num_of_cpu)
            while True:
                mul_result = [pool.apply_async(self.run_simulation, args=(game, wins, plays, scores)) for _ in range(self.num_of_cpu)]
                results = [res.get() for res in mul_result]

                for tmp_wins, tmp_plays, tmp_scores in results:
                    for k, v in tmp_wins.items():
                        wins[k] += v

                    for k, v in tmp_plays.items():
                        plays[k] += v

                    for k, v in tmp_scores.items():
                        scores[k] += v

                count_simulation += self.num_of_cpu

                if time.time() - stime >= simulation_time_limit:
                    break

            pool.close()

            moves_states = []
            for card in valid_cards:
                seen_cards = [c for c in self.seen_cards[:]]
                seen_cards.append(card)

                moves_states.append((card, tuple(seen_cards)))

            tmp_wins = []
            percent_wins, average_score, played_rank_card, played_score_card = -sys.maxsize, sys.maxsize, None, None
            for card, states in moves_states:
                k = (game.current_player_idx, states)

                avg_score =  scores[k] / plays[k]
                if avg_score < average_score:
                    average_score = avg_score
                    played_score_card = card

                win_rate = -1
                if k in wins:
                    win_rate = wins[k] / max(plays[k], 1)
                    tmp_wins.append(win_rate)

                    if win_rate > percent_wins:
                        percent_wins = win_rate
                        played_rank_card = card
                else:
                    print("not found", k, "current win_rate is", percent_wins)

                print("{}: ranking ---> {:.2f}, score ---> {:.2f}".format(card, win_rate, avg_score))

            played_card = played_rank_card

            sum_win_rate = sum(tmp_wins)
            if sum_win_rate == 0:
                """
                percent_wins, played_card = sys.maxsize, None
                for card, states in moves_states:
                    k = (game.current_player_idx, states)

                    avg_score = scores[k] / plays[k]
                    #print(card, "score --->", k, avg_score)
                    if avg_score < percent_wins:
                        percent_wins = avg_score
                        played_card = card
                """
                played_card = played_score_card

            print("(is_ranking_mode, count_simulation, wins, plays, scores, played_rank_card, played_score_card, valid_cards) = ({}, {}, {}, {}, {}, {}({:.2f}), {}({:.2f}), {})".format(\
                sum_win_rate > 0, count_simulation, len(wins), len(plays), len(scores), played_rank_card, percent_wins, played_score_card, average_score, valid_cards))
        else:
            played_card = valid_cards[0]

            if self.verbose:
                print("don't need simulation, can only play {} card".format(played_card))

        return played_card


    def run_simulation(self, game, wins, plays, scores):
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
        winners = None
        tmp_plays, tmp_wins, tmp_scores = {}, {}, {}

        init_card = None
        for t in range(self.max_moves):
            player_idx = game.current_player_idx
            player = game.players[player_idx]

            played_card, state = None, None
            if player_idx == self.position:
                #stime = time.time()
                valid_cards = player.get_valid_cards(game._player_hands[game.current_player_idx], game.trick, game.trick_nr, game.is_heart_broken)

                is_all_pass, log_total = True, 0
                moves_states = []
                for card in valid_cards:
                    seen_cards = [c for c in player.seen_cards[:]] + [card]

                    key = (card, tuple(seen_cards))
                    moves_states.append(key)

                    is_all_pass &= key in plays
                    if is_all_pass:
                        log_total += plays[key]

                if is_all_pass:
                    log_total = np.log(log_total) #np.log(sum(plays[(player_idx, state)] for card, state in moves_states))

                    value, played_card, state = max(
                        ((wins[(player_idx, state)] / plays[(player_idx, state)]) + self.C * np.sqrt(log_total / plays[(player_idx, state)]),
                          card,
                          state)
                        for card, state in moves_states)
                else:
                    if t == 0:
                        played_card = choice(valid_cards)
                    else:
                        played_card = player.play_card(game._player_hands[player_idx], game)

                    state = tuple(game.players[0].seen_cards[:]) + (played_card,)

                #if played_card in [Card(Suit.spades, Rank.two), Card(Suit.spades, Rank.seven), Card(Suit.spades, Rank.ten)]:
                #    print(played_card, state)
                #print(time.time()-stime, len(moves_states))
            else:
                played_card = player.play_card(game._player_hands[player_idx], game)

            if t == 0:
                init_card = played_card

            game.step(played_card)

            if player_idx == self.position:
                if expand and (player_idx, state) not in plays:
                    expand = False

                    tmp_plays[(player_idx, state)] = 0
                    tmp_wins[(player_idx, state)] = 0
                    tmp_scores[(player_idx, state)] = 0

                visited_states.add(state)

            winners = game.get_game_winners()
            if winners:
                if self.position in winners:
                    print(init_card, "get winners", winners, game.player_scores)
                else:
                    print(init_card, "not winners", winners, game.player_scores)

                break

        for state in visited_states:
            key = (self.position, state)
            if key in tmp_plays or key in plays:
                tmp_plays.setdefault(key, 0)
                tmp_plays[key] += 1

                tmp_scores.setdefault(key, 0)
                tmp_scores[key] += game.player_scores[self.position]

                if self.position in winners:
                    tmp_wins.setdefault(key, 0)
                    tmp_wins[key] += 1

                    #print("key >>>>>>", key, tmp_wins[key], winners)
            else:
                print("not found", key, init_card)

        return tmp_wins, tmp_plays, tmp_scores
