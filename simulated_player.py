"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np
import multiprocessing as mp

from collections import defaultdict
from random import shuffle, choice, randint, choice

from card import Suit, Rank, Card, Deck
from rules import is_card_valid

from player import Player, SimplePlayer, StupidPlayer

TIMEOUT_SECOND = 0.9
COUNT_CPU = mp.cpu_count()

class MonteCarloPlayer(SimplePlayer):
    def __init__(self, num_of_cpu=COUNT_CPU, verbose=False):
        super(MonteCarloPlayer, self).__init__(verbose=verbose)

        self.num_of_cpu = num_of_cpu


    def play_card(self, hand_cards, game, simulation_time_limit=TIMEOUT_SECOND):
        valid_cards = self.get_valid_cards(hand_cards, game.trick, game.trick_nr, game.is_heart_broken)

        card = None
        if len(valid_cards) > 1:
            winning_rate = defaultdict(int)
            winning_score = defaultdict(int)

            stime = time.time()

            count_simulation = 0

            is_score = False
            pool = mp.Pool(processes=self.num_of_cpu)
            while True:
                mul_result = [pool.apply_async(self.run_simulation, args=(idx, copy.deepcopy(game), hand_cards, card)) for idx, card in enumerate(valid_cards)]
                results = [res.get() for res in mul_result]

                for idx, score, is_winner in results:
                    winning_score[idx] += -score
                    winning_rate[idx] += 1 if is_winner else 0

                    count_simulation += 1

                if time.time() - stime >= simulation_time_limit:
                    break

            pool.close()

            is_score = all([v == 0 for v in winning_rate.values()])

            rating = winning_rate
            if is_score:
                rating = winning_score

            pig_winning_rate = None
            simulation_rate = None
            for card_idx, c in sorted(rating.items(), key=lambda x: x[1]/count_simulation):
                card = valid_cards[card_idx]
                simulation_rate = abs(c/count_simulation)

                if card == Card(Suit.spades, Rank.queen):
                    pig_winning_rate = abs(c/count_simulation)

                if self.verbose:
                    print("simulation: {} round".format(game.trick_nr), "valid_cards: {}, simulate {} card --> {:3d} {} /{:4d} wins {:.4f}".format(\
                        valid_cards, card, abs(c), "score" if is_score else "rounds", count_simulation, abs(c/count_simulation)))

            if pig_winning_rate is not None and card != Card(Suit.spades, Rank.queen):
                if is_score:
                    if abs(-pig_winning_rate+simulation_rate) < 2:
                        card = Card(Suit.spades, Rank.queen)

                        if self.verbose:
                            self.say("score mode: force to play QS({} vs. {})", -pig_winning_rate, -simulation_rate)
                elif not is_score:
                    if abs(pig_winning_rate-simulation_rate) < 0.001:
                        card = Card(Suit.spades, Rank.queen)

                        if self.verbose:
                            self.say("rate mode: force to play QS({} vs. {})", pig_winning_rate, simulation_rate)
        else:
            card = valid_cards[0]

            if self.verbose:
                print("don't need simulation, can only play {} card".format(card))

        return card


    def get_remaining_cards(self, hand_cards):
        deck = Deck()

        remaining_cards = []
        for c in deck.cards:
            for pc in hand_cards + self.seen_cards:
                if c == pc:
                    break
            else:
                remaining_cards.append(c)

        return remaining_cards


    def redistribute_cards(self, game, remaining_cards):
        shuffle(remaining_cards)

        for idx in range(len(game._player_hands)):
            if idx != self.position:
                game._player_hands[idx] = np.random.choice(remaining_cards, len(game._player_hands[idx]), replace=False).tolist()

                for used_card in game._player_hands[idx]:
                    remaining_cards.remove(used_card)

        if remaining_cards:
            print("error", type(self).__name__, [len(v) for v in game._player_hands])
            raise


    def run_simulation(self, idx, game, hand_cards, played_card):
        remaining_cards = self.get_remaining_cards(hand_cards)

        game.verbose = False
        game.players = [SimplePlayer(), SimplePlayer(), SimplePlayer(), SimplePlayer()]

        self.redistribute_cards(game, remaining_cards[:])

        game.step(played_card)

        for _ in range(4-len(game.trick)):
            game.step()

        #winning_card = game.round_over()

        for _ in range(13-game.trick_nr):
            game.play_trick()

        game.score()

        min_score = min([score for score in game.player_scores])
        player_score = game.player_scores[self.position]

        #print("win??", self.position,  game.player_scores, player_score, min_score)

        return idx, player_score, player_score == min_score


class MonteCarloPlayer2(MonteCarloPlayer):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer2, self).__init__(verbose=verbose)


    def redistribute_cards(self, game, remaining_cards):
        shuffle(remaining_cards)

        ori_size = []
        for idx in range(len(game._player_hands)):
            if idx != self.position:
                size = len(game._player_hands[idx])
                ori_size.append(size)

                game._player_hands[idx] = []

                removed_cards = []
                for card in remaining_cards:
                    if game.lacking_cards[idx][card.suit] == False:
                        game._player_hands[idx].append(card)
                        removed_cards.append(card)

                        if len(game._player_hands[idx]) == size:
                            break

                for card in removed_cards:
                    remaining_cards.remove(card)
            else:
                ori_size.append(len(game._player_hands[idx]))


        lacking_idx = [idx for idx in range(4) if len(game._player_hands[idx]) != ori_size[idx]]
        while remaining_cards:
            for card in remaining_cards:
                latest_lacking_idx = [idx for idx in range(4) if len(game._player_hands[idx]) != ori_size[idx]]

                player_idx = lacking_idx[0]
                while player_idx in lacking_idx or player_idx == self.position:
                    player_idx = randint(0, 3)

                hand_cards = game._player_hands[player_idx]

                for card_idx, hand_card in enumerate(hand_cards):
                    if not game.lacking_cards[latest_lacking_idx[0]][hand_card.suit]:
                        game._player_hands[player_idx][card_idx] = card
                        game._player_hands[latest_lacking_idx[0]].append(hand_card)
                        remaining_cards.remove(card)

                        break

        if remaining_cards:
            print("error", remaining_cards)
            for lacking_card_info, hand_cards in zip(game.lacking_cards, game._player_hands):
                print(lacking_card_info, hand_cards, len(hand_cards))
