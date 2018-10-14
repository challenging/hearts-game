import sys

import time

import numpy as np
import multiprocessing as mp

from functools import cmp_to_key
from random import shuffle
from collections import defaultdict

from card import Deck, Suit, Rank, Card
from card import card_to_bitmask

from simulated_player import MonteCarloPlayer5, COUNT_CPU

from simple_complex_game import run_simulation
from strategy_play import random_choose, greedy_choose
from expert_play import expert_choose


TIMEOUT_SECOND = 0.90


def sorted_suits(xs, ys):
    if len(xs[1]) == len(ys[1]):
        if ys[0] == Suit.spades:
            return 5 - xs[0].value
        else:
           return ys[0].value - xs[0].value
    else:
        if ys[0] == Suit.spades:
            return sys.maxsize
        elif ys[0] == Suit.hearts:
            return sys.maxsize-1
        else:
            return len(xs[1]) - len(ys[1])


class MonteCarloPlayer7(MonteCarloPlayer5):
    def __init__(self, num_of_cpu=COUNT_CPU, verbose=False):
        super(MonteCarloPlayer7, self).__init__(verbose=verbose)


    def set_proactive_mode(self, hand, round_idx):
        hand.sort(key=lambda x: self.undesirability(x), reverse=True)

        hand_cards = {Suit.spades: [], Suit.hearts: [], Suit.diamonds: [], Suit.clubs: []}
        for card in hand:
            hand_cards[card.suit].append(max(card.rank.value-10, 0))

        pass_low_card = False

        point_of_suit = 0
        for suit, cards in hand_cards.items():
            point_of_suit = np.sum(cards)
            if suit == Suit.hearts:
                if (point_of_suit > 7 and len(cards) > 5):
                    self.proactive_mode.add(suit)
            else:
                if (point_of_suit > 6 and len(cards) > 4) and (len(hand_cards[Suit.hearts]) > 2 and np.sum(hand_cards[Suit.hearts]) > 3):
                    self.proactive_mode.add(suit)
                elif (point_of_suit > 5 and len(cards) > 5) and (len(hand_cards[Suit.hearts]) > 2 and np.sum(hand_cards[Suit.hearts]) > 3):
                    self.proactive_mode.add(suit)
                elif (point_of_suit > 4 and len(cards) > 6) and (len(hand_cards[Suit.hearts]) > 2 and np.sum(hand_cards[Suit.hearts]) > 3):
                    self.proactive_mode.add(suit)

        if not self.proactive_mode:
            points = np.sum([v for vv in hand_cards.values() for v in vv])

            if points > 15 and np.sum(hand_cards[Suit.hearts]) > 3:
                pass_low_card = True

        return hand_cards, pass_low_card


    def pass_cards(self, hand, round_idx):
        hand_cards, pass_low_card = self.set_proactive_mode(hand, round_idx)

        pass_cards, keeping_cards = [], []
        if self.proactive_mode:
            for card in hand:
                if card.suit not in self.proactive_mode:
                    if card.suit != Suit.hearts:
                        pass_cards.append(card)
                    else:
                        if card.rank < Rank.jack:
                            pass_cards.append(card)

            pass_cards.sort(key=lambda x: self.undesirability(x), reverse=False)

            if len(pass_cards) < 3:
                hand.sort(key=lambda x: self.undesirability(x), reverse=False)

                pass_cards.extend(hand[:3-len(pass_cards)])

            self.say("{} ----> proactive_mode: {}, pass cards are {}, hand_cards are {}",\
                 type(self).__name__, self.proactive_mode, pass_cards[:3], hand)
        elif pass_low_card:
            hand.sort(key=lambda x: self.undesirability(x), reverse=False)
            pass_cards = hand

            self.say("{} ----> pass low cards are {} from {}({})", type(self).__name__, pass_cards[:3], hand, hand_cards)
        else:
            num_of_suits = defaultdict(list)
            for card in hand:
                num_of_suits[card.suit].append(card)

            if len(num_of_suits.get(Suit.spades, [])) < 5:
                for card in num_of_suits.get(Suit.spades, []):
                    if card.rank >= Rank.queen:
                        pass_cards.append(card)

            if len(num_of_suits.get(Suit.hearts, [])) < 5:
                for card in num_of_suits.get(Suit.hearts, []):
                    if card.rank >= Rank.queen:
                        pass_cards.append(card)


            if len(pass_cards) < 2:
                for suit, cards in sorted(num_of_suits.items(), key=cmp_to_key(sorted_suits)):
                    if len(pass_cards) >= 3:
                        break

                    if len(cards) > 2:
                        continue

                    if cards[0].suit in [Suit.clubs, Suit.diamonds]:
                        if len(cards) == 2 and len(pass_cards) == 2:
                            continue

                        for card in sorted(cards, key=lambda x: -x.rank.value):
                            if card not in pass_cards: pass_cards.append(card)
                    elif cards[0].suit == Suit.spades:
                        for card in cards:
                            if card.rank >= Rank.queen and card not in pass_cards:
                                if card not in pass_cards: pass_cards.append(card)
                    elif cards[0].suit == Suit.hearts:
                        for card in cards:
                            if card.rank >= Rank.jack:
                                if card not in pass_cards: pass_cards.append(card)

            if len(pass_cards) < 3:
                hand.sort(key=lambda x: self.undesirability(x), reverse=True)
                pass_cards.extend([card for card in hand if card not in pass_cards])

                self.say("{} ----> pass undesirability cards are {} from {}({})", type(self).__name__, pass_cards[:3], hand, hand_cards)
            else:
                self.say("{} ----> pass short cards are {} from {}({})", type(self).__name__, pass_cards[:3], hand, hand_cards)

        self.say("proactive mode: {}, keeping_cards are {}, pass card is {}", self.proactive_mode, keeping_cards, pass_cards[:3])

        return pass_cards[:3]


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        stime = time.time()

        game.are_hearts_broken()

        for player_idx, suits in other_info.get("lacking_info", {}).items():
            for suit in suits:
                game.lacking_cards[player_idx][suit] = True

            self.say("Player-{} may lack of {} suit({}, {})", player_idx, suit, game.lacking_cards[player_idx], other_info)

        self.say("Player-{}, the information of lacking_card is {}", self.position, [(player_idx, k) for player_idx, info in enumerate(game.lacking_cards) for k, v in info.items() if v])

        hand_cards = [[] if player_idx != self.position else game._player_hands[player_idx] for player_idx in range(4)]

        remaining_cards = Deck().cards
        for card in self.seen_cards + hand_cards[self.position]:
            remaining_cards.remove(card)

        taken_cards = []
        for player_idx, cards in enumerate(game._cards_taken):
            taken_cards.append(card_to_bitmask(cards))

        init_trick = [[None, game.trick]]

        void_info = {}
        for player_idx, info in enumerate(game.lacking_cards):
            if player_idx != self.position:
                void_info[player_idx] = info

        must_have = self.transfer_cards

        played_card = None

        selection_func = [expert_choose, greedy_choose]
        self.say("proactive_mode: {}, selection_func={}, num_of_cpu={}, is_heart_broken={}, expose_heart_ace={}", \
            self.proactive_mode, selection_func, self.num_of_cpu, game.is_heart_broken, game.expose_heart_ace)

        pool = mp.Pool(processes=self.num_of_cpu)
        mul_result = [pool.apply_async(run_simulation, args=(seed,
                                                             game.trick_nr+1, 
                                                             self.position, 
                                                             init_trick, 
                                                             hand_cards, 
                                                             game.is_heart_broken, 
                                                             game.expose_heart_ace, 
                                                             remaining_cards, 
                                                             taken_cards, 
                                                             played_card, 
                                                             selection_func, 
                                                             must_have, 
                                                             void_info, 
                                                             None,
                                                             TIMEOUT_SECOND-0.02)) for seed in range(self.num_of_cpu)]

        partial_results = [res.get() for res in mul_result]

        results = defaultdict(list)
        for row in partial_results:
            for card, info in row.items():
                results[card].extend(info)

        pool.close()

        min_score = sys.maxsize
        for card, info in results.items():
            total_size, total_score, total_num = 0, 0, 0

            for score, self_shoot_the_moon in info:
                total_score += score
                total_num += 1 if self_shoot_the_moon else 0
                total_size += 1

            mean_score = total_score / len(info)

            if mean_score < min_score:
                played_card = card
                min_score = mean_score

            ma, mi = -sys.maxsize, sys.maxsize
            for score, _ in info:
                if score > ma:
                    ma = score
                if score < mi:
                    mi = score

            self.say("simulate {} card with {:4d} times, and get {:.3f} score ({:.4f} ~ {:.4f}), num_moon={}, {:.2f}%", \
                card, total_size, mean_score, ma, mi, total_num, 100.0*total_num/total_size)

        self.say("pick {} card, cost {:.8} seconds", played_card, time.time()-stime)

        return played_card
