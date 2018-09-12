"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np
import multiprocessing as mp

from scipy.stats import describe
from collections import defaultdict
from random import shuffle, choice

from card import Suit, Rank, Card, Deck
from rules import is_card_valid

from simulated_player import MonteCarloPlayer5
from player import StupidPlayer, SimplePlayer

TIMEOUT_SECOND = 0.9
COUNT_CPU = mp.cpu_count()


class MCTSPlayer(MonteCarloPlayer5):
    def __init__(self, C, verbose=False):
        super(MCTSPlayer, self).__init__(verbose=verbose)

        self.C = C


    def play_card(self, game, simulation_time_limit=TIMEOUT_SECOND):
        game.are_hearts_broken()

        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        card = None
        if len(valid_cards) > 1:
            stime = time.time()

            scores = defaultdict(list)
            pool = mp.Pool(processes=self.num_of_cpu)
            while time.time() - stime < simulation_time_limit:
                mul_result = [pool.apply_async(self.run_simulation, args=(game, scores)) for _ in range(self.num_of_cpu)]
                results = [res.get() for res in mul_result]

                for tmp_scores in results:
                    for k, v in tmp_scores.items():
                        scores[k].append(v)
            pool.close()

            if Suit.hearts in self.proactive_mode:
                deck = Deck()

                remaining_cards = {}
                for card in deck.cards:
                    if card.suit == Suit.hearts and card not in self.seen_cards and card not in game.trick and card not in game._player_hands[self.position]:
                        remaining_cards.setdefault(card.suit, card)
                        if card.rank > remaining_cards[card.suit].rank:
                            remaining_cards[card.suit] = card

                for card in valid_cards[::-1]:
                    if card.suit in remaining_cards:
                        if card > remaining_cards[card.suit]:
                            self.say("1. force to get this card - {} from {} because of {}", card, valid_cards, game.trick)

                            return card


            if game.trick and Suit.hearts in self.proactive_mode:
                is_point_card_in_trick, leading_suit, current_max_rank = False, game.trick[0].suit, None
                for card in game.trick:
                    if current_max_rank is None:
                        current_max_rank = card.rank
                    else:
                        if card.suit == leading_suit and card.rank > current_max_rank:
                            current_max_rank = card.rank

                        if card.suit == Suit.hearts:
                            is_point_card_in_trick = True

                if is_point_card_in_trick and valid_cards[-1].suit == leading_suit and valid_cards[-1].rank > current_max_rank:
                    self.say("2. force to get this card - {} from {} because of {}", valid_cards[-1], valid_cards, game.trick)

                    return valid_cards[-1]


            moves_states = []
            for card in valid_cards:
                seen_cards = [c for c in self.seen_cards[:]] + [card]
                moves_states.append((card, tuple(seen_cards)))

            average_score, most_visit, played_card = sys.maxsize, -sys.maxsize, None
            for state in moves_states:
                if state in scores:
                    avg_score = np.mean(scores[state])
                    n_visit = len(scores[state])

                    if n_visit > most_visit:
                        most_visit = n_visit
                        played_card = state[0]

                    statess = describe(scores[state])
                    self.say("{}, pick {}: (n={}, mean={:.4f}, std={:.4f}, minmax={}", valid_cards, state[0], statess.nobs, statess.mean, statess.variance**0.5, statess.minmax)
                else:
                    self.say("not found {} in {}", state)

            self.say("(scores, played_card, valid_cards) = ({}, {}({:.2f}), {})",
                len(scores), played_card, most_visit, valid_cards)
        else:
            played_card = self.no_choice(valid_cards[0])
            #self.say("don't need simulation, can only play {} card", played_card)

        return played_card


    def run_simulation(self, game, scores):
        hand_cards = game._player_hands[game.current_player_idx]
        remaining_cards = self.get_remaining_cards(hand_cards)

        seen_cards = copy.deepcopy(game.players[0].seen_cards)
        game.verbose = False
        game.players = self.get_players(game)#[SimplePlayer() for idx in range(4)]
        for player in game.players:
            player.seen_cards = copy.deepcopy(seen_cards)

        trick_nr = game.trick_nr

        game = self.redistribute_cards(game, remaining_cards[:])

        tmp_plays, tmp_scores = {}, defaultdict(float)

        player = game.players[self.position]

        played_card = None
        valid_cards = player.get_valid_cards(game._player_hands[self.position], game)

        is_all_pass, log_total = True, 0
        moves_states = []
        for card in valid_cards:
            tmp_seen_cards = [c for c in seen_cards[:]] + [card]

            key = (card, tuple(tmp_seen_cards))
            moves_states.append(key)

            is_all_pass &= (key in scores)
            if is_all_pass:
                log_total += len(scores[key])

        if is_all_pass:
            log_total = np.log(log_total)

            value, state = max(
                (-np.mean(scores[state]) + self.C * np.sqrt(log_total / len(scores[state])), state) for state in moves_states)

            played_card = state[0]
        else:
            played_card = choice(valid_cards)

        state = (played_card, tuple(game.players[0].seen_cards[:]) + (played_card,))

        game.step(played_card)

        for _ in range(4-len(game.trick)):
            game.step()

        for _ in range(13-game.trick_nr):
            game.play_trick()

        if trick_nr < 6:
            self.overwrite_game_rule(trick_nr, game)

        game.score()

        tmp_scores[state] = self.score_func(game.player_scores)

        return tmp_scores
