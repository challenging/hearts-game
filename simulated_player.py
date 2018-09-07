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
from player import StupidPlayer, SimplePlayer


TIMEOUT_SECOND = 0.9
COUNT_CPU = mp.cpu_count()

class MonteCarloPlayer(SimplePlayer):
    def __init__(self, num_of_cpu=COUNT_CPU, verbose=False):
        super(MonteCarloPlayer, self).__init__(verbose=verbose)

        self.num_of_cpu = num_of_cpu


    def pass_cards(self, hand, round_idx):
        hand.sort(key=lambda x: self.undesirability(x), reverse=True)

        hand_cards = {Suit.spades: [], Suit.hearts: [], Suit.diamonds: [], Suit.clubs: []}
        for card in hand:
            hand_cards[card.suit].append(max(card.rank.value-10, 0))

        proactive_mode, proactive_suit = False, None
        for suit, cards in hand_cards.items():
            if suit == Suit.hearts:
                if np.sum(cards) >= 8 and len(cards) >= 4:
                    proactive_mode = True
                elif np.sum(cards) >= 7 and len(cards) >= 5:
                    proactive_mode = True
            else:
                if np.sum(cards) >= 6 and len(cards) >=5:
                    if len(hand_cards[Suit.hearts]) >= 3 and np.sum(hand_cards[Suit.hearts]) >= 1:
                        proactive_mode = True
                        proactive_suit = suit

        pass_cards = []
        if proactive_mode:
            for card in hand:
                if card.suit not in [Suit.hearts, proactive_suit]:
                    pass_cards.append(card)
            pass_cards.sort(key=lambda x: self.undesirability(x), reverse=False)

            self.say("{} ----> proactive_mode: {}".format(type(self).__name__, proactive_mode), pass_cards[:3], sorted(hand))
        else:
            pass_cards = []
            if Card(Suit.spades, Rank.queen) in hand:
                num_spades = 0
                for card in hand:
                    if card.suit == Suit.spades:
                        if card.rank < Rank.queen:
                            num_spades += 1

                if num_spades > 2:
                    for card in hand:
                        if card.suit != Suit.spades:
                            pass_cards.append(card)

            if not pass_cards:
                pass_cards = hand
                pass_cards.sort(key=lambda x: self.undesirability(x), reverse=True)

        self.say("pass card is {}", pass_cards[:3])

        return pass_cards[:3]


    def winning_score_func(self, scores):
        return np.mean(scores)


    def play_card(self, hand_cards, game, simulation_time_limit=TIMEOUT_SECOND):
        game.are_hearts_broken()
        valid_cards = self.get_valid_cards(hand_cards, game)

        card = None
        if len(valid_cards) > 1:
            stime = time.time()

            winning_score = defaultdict(list)
            pool = mp.Pool(processes=self.num_of_cpu)
            while time.time() - stime < simulation_time_limit:
                mul_result = [pool.apply_async(self.run_simulation, args=(game, hand_cards, card)) for card in valid_cards]
                results = [res.get() for res in mul_result]

                for card, score in results:
                    winning_score[card].append(score)
            pool.close()

            min_score = sys.maxsize
            for card in valid_cards:
                score = self.winning_score_func(winning_score[card])

                if score < min_score:
                    min_score = score
                    played_card = card

                self.say("{} {}, simulation: {} round --> valid_cards: {}, simulate {} card --> average_score {:.4f} --> {:.4f}, ({})",
                    game.trick_nr,
                    type(self).__name__,
                    len(winning_score[card]),
                    valid_cards,
                    card,
                    np.mean(winning_score[card]),
                    score,
                    describe(winning_score[card]))
        else:
            played_card = valid_cards[0]

        self.say("pick {} card, {}", played_card, game.expose_heart_ace)
        return played_card


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
            self.say("Error in redistributing cards, {}, {}, {}", type(self).__name__, remaining_cards, [len(v) for v in game._player_hands])
            raise

        return game


    def redistribute_cards(self, game, remaining_cards):
        return self.simple_redistribute_cards(game, remaining_cards)


    def score_func(self, scores):
        return scores[self.position]


    def run_simulation(self, game, hand_cards, played_card):
        remaining_cards = self.get_remaining_cards(hand_cards)

        game.verbose = False
        game.players = [SimplePlayer() for _ in range(4)]

        game = self.redistribute_cards(game, remaining_cards[:])

        game.step(played_card)

        for _ in range(4-len(game.trick)):
            game.step()

        for _ in range(13-game.trick_nr):
            game.play_trick()

        game.score()

        self_score = self.score_func(game.player_scores)
        #print(played_card, self_score, game.player_scores)

        return played_card, self_score


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

        return copy_game


class MonteCarloPlayer3(MonteCarloPlayer2):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer3, self).__init__(verbose=verbose)


    def score_func(self, scores):
        min_score, other_score = None, 0
        for idx, score in enumerate(sorted(scores)):
            if idx == 0:
                min_score = score
            else:
                other_score += score

        self_score = scores[self.position]
        if self_score == min_score:
            return self_score-other_score/3
        else:
            return self_score-min_score


class MonteCarloPlayer4(MonteCarloPlayer3):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer4, self).__init__(verbose=verbose)


    def overwrite_game_score_func(self, game):
        import types

        def score(self):
            self.check_shootmoon()

            if self.is_shootmoon:
                max_score = max([self.count_points(self._cards_taken[idx]) for idx in range(4)])

                for i in range(4):
                    s = self.count_points(self._cards_taken[i])

                    if s > 0:
                        self.player_scores[i] = 0
                    else:
                        self.player_scores[i] = max_score
            else:
                for i in range(4):
                    self.player_scores[i] = self.count_points(self._cards_taken[i])
                    if self.player_scores[i] == 0:
                        for card in self._cards_taken[i]:
                            if card == Card(Suit.clubs, Rank.ten):
                                self.player_scores[i] = 2

                                break

        game.score = types.MethodType(score, game)


    def run_simulation(self, game, hand_cards, played_card):
        remaining_cards = self.get_remaining_cards(hand_cards)

        trick_nr = game.trick_nr

        game.verbose = False
        game.players = [SimplePlayer() for _ in range(4)]

        game = self.redistribute_cards(game, remaining_cards)

        game.step(played_card)

        for _ in range(4-len(game.trick)):
            game.step()

        for _ in range(13-game.trick_nr):
            game.play_trick()

        if trick_nr < 6:
            self.overwrite_game_score_func(game)

        game.score()

        self_score = self.score_func(game.player_scores)
        #print(played_card, self_score, game.player_scores)

        return played_card, self_score


class MonteCarloPlayer5(MonteCarloPlayer4):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer5, self).__init__(verbose=verbose)


    def run_simulation(self, game, hand_cards, played_card):
        remaining_cards = self.get_remaining_cards(hand_cards)

        trick_nr = game.trick_nr

        game.verbose = False
        game.players = [SimplePlayer() for _ in range(4)]

        game = self.redistribute_cards(game, remaining_cards)
        #print("&&&&&&", game._player_hands)

        game.step(played_card)

        for _ in range(4-len(game.trick)):
            game.step()

        for _ in range(13-game.trick_nr):
            game.play_trick()

        if trick_nr < 6:
            self.overwrite_game_score_func(game)

        game.score()

        self_score = self.score_func(game.player_scores)
        #print(game.point_cards)
        #print("******", played_card, self_score, game.player_scores, [[card for card in cards if card in game.point_cards] for cards in game._cards_taken])

        return played_card, self_score

    def winning_score_func(self, scores):
        stats = describe(scores)

        additional_score = 0.2*stats.variance**0.5
        #if stats.skewness+1e-16 < 0:
        #    additional_score = 0.1*stats.variance**0.5
        #    print("****notice****")

        return np.mean(scores)+additional_score
