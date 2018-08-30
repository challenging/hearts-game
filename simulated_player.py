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
            winning_score = defaultdict(int)

            stime = time.time()

            count_simulation = defaultdict(int)

            pool = mp.Pool(processes=self.num_of_cpu)
            while time.time() - stime < simulation_time_limit:
                mul_result = [pool.apply_async(self.run_simulation, args=(idx, copy.deepcopy(game), hand_cards, card)) for idx, card in enumerate(valid_cards)]
                results = [res.get() for res in mul_result]

                for idx, score in results:
                    winning_score[idx] += score
                    count_simulation[idx] += 1

            pool.close()

            for card_idx, c in sorted(winning_score.items(), key=lambda x: -x[1]/count_simulation[x[0]]):
                card = valid_cards[card_idx]
                winning_rate = c/count_simulation[card_idx]

                self.say("{}, simulation: {} round --> valid_cards: {}, simulate {} card --> {:3d} score /{:4d} average_score {:.4f}".format(\
                    type(self).__name__, game.trick_nr, valid_cards, card, c, count_simulation[card_idx], winning_rate))
        else:
            card = valid_cards[0]

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


    def simple_redistribute_cards(self, game, remaining_cards):
        shuffle(remaining_cards)

        for idx in range(len(game._player_hands)):
            if idx != self.position:
                game._player_hands[idx] = np.random.choice(remaining_cards, len(game._player_hands[idx]), replace=False).tolist()

                for used_card in game._player_hands[idx]:
                    remaining_cards.remove(used_card)

        if remaining_cards:
            print("error", type(self).__name__, remaining_cards, [len(v) for v in game._player_hands])
            raise

    def redistribute_cards(self, game, remaining_cards):
        self.simple_redistribute_cards(game, remaining_cards)


    def score_func(self, scores):
        return scores[self.position]


    def run_simulation(self, idx, game, hand_cards, played_card):
        remaining_cards = self.get_remaining_cards(hand_cards)

        game.verbose = False
        game.players = [StupidPlayer() for _ in range(4)]

        self.redistribute_cards(game, remaining_cards[:])

        game.step(played_card)

        for _ in range(4-len(game.trick)):
            game.step()

        for _ in range(13-game.trick_nr):
            game.play_trick()

        game.score()

        return idx, self.score_func(game.player_scores)


class MonteCarloPlayer2(MonteCarloPlayer):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer2, self).__init__(verbose=verbose)


    def redistribute_cards(self, copy_game, copy_remaining_cards):
        retry = 3
        while retry >= 0:
            game = copy.deepcopy(copy_game)
            remaining_cards = copy.deepcopy(copy_remaining_cards)

            shuffle(remaining_cards)

            ori_size, fixed_cards = [], set()
            for idx in range(len(game._player_hands)):
                if idx != self.position:
                    size = len(game._player_hands[idx])
                    ori_size.append(size)

                    game._player_hands[idx] = []

                    for card in self.transfer_cards.get(idx, []):
                        if card in remaining_cards:
                            game._player_hands[idx].append(card)
                            remaining_cards.remove(card)

                            fixed_cards.add(card)

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


            retry2 = 3
            lacking_idx = [idx for idx in range(4) if len(game._player_hands[idx]) < ori_size[idx]]
            while retry2 >= 0 and any([ori_size[player_idx] != len(game._player_hands[player_idx]) for player_idx in range(4)]):
                removed_cards = []
                for card in remaining_cards:
                    latest_lacking_idx = [idx for idx in range(4) if len(game._player_hands[idx]) < ori_size[idx]]

                    player_idx = choice([player_idx for player_idx in range(4) if player_idx not in latest_lacking_idx + [self.position]])
                    hand_cards = game._player_hands[player_idx]

                    for card_idx, hand_card in enumerate(hand_cards):
                        if hand_card not in fixed_cards and not game.lacking_cards[latest_lacking_idx[0]][hand_card.suit]:
                            game._player_hands[player_idx][card_idx] = card
                            game._player_hands[latest_lacking_idx[0]].append(hand_card)

                            removed_cards.append(card)

                            break

                for card in removed_cards:
                    remaining_cards.remove(card)

                for player_idx, size in enumerate(ori_size):
                    if len(game._player_hands[player_idx]) > size:
                        remaining_cards.extend(game._player_hands[player_idx][size:])
                        game._player_hands[player_idx] = game._player_hands[player_idx][:size]

                retry2 -= 1

            if remaining_cards or any([ori_size[player_idx] != len(game._player_hands[player_idx]) for player_idx in range(4)]):
                self.say("error --> remaining_cards: {}, ori_size:{}, simulated_hand_cards: {}", remaining_cards, ori_size, game._player_hands)

                retry -= 1
            else:
                copy_game = game

                break
        else:
            self.say("apply self.simple_redistribute_cards")

            self.simple_redistribute_cards(copy_game, copy_remaining_cards)


class MonteCarloPlayer3(MonteCarloPlayer2):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer2, self).__init__(verbose=verbose)


    def score_func(self, scores):
        min_score, second_score = None, None
        for idx, score in enumerate(sorted(scores)):
            if idx == 0:
                min_score = score
            elif idx == 1:
                second_score = score
                break

        self_score = scores[self.position]
        if self_score == min_score:
            return self_score-second_score
        else:
            return self_score-min_score
