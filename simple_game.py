#!/usr/bin/env python

import sys

import time
import copy

import numpy as np

from collections import defaultdict

from card import Rank, Suit, Card, Deck

from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import EMPTY_CARDS, FULL_CARDS
from card import card_to_bitmask, str_to_bitmask, count_points

from rules import is_card_valid, transform
from redistribute_cards import redistribute_cards
from strategy_play import random_choose, greedy_choose, expert_choose


IS_DEBUG = False


class SimpleGame(object):
    def __init__(self,
                 position,
                 hand_cards,
                 score_cards=None,
                 is_hearts_borken=False,
                 is_show_pig_card=False,
                 is_show_double_card=False,
                 is_shoot_the_moon=True,
                 expose_hearts_ace=False,
                 tricks=[]):

        self.position = position
        self.hand_cards = hand_cards

        self.is_show_pig_card = is_show_pig_card
        self.is_show_double_card = is_show_double_card
        self.is_shoot_the_moon = is_shoot_the_moon

        if score_cards is None:
            self.score_cards = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        else:
            self.score_cards = score_cards

            if not self.is_show_pig_card or not self.is_show_double_card:
                for cards in self.score_cards:
                    if self.is_show_pig_card and self.is_show_double_card:
                        break
                    else:
                        self.check_show_pig_card(cards)
                        self.check_show_double_card(cards)

            if not self.is_shoot_the_moon:
                self.check_shoot_the_moon(self.score_cards)

        self.tricks = tricks

        self.is_hearts_broken = is_hearts_borken
        self.expose_hearts_ace = 2 if expose_hearts_ace else 1

        self.played_card = None


    def check_show_pig_card(self, cards):
        self.is_show_pig_card = (cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["Q"] > 0)


    def check_show_double_card(self, cards):
        self.is_show_double_card = (cards[SUIT_TO_INDEX["C"]] & NUM_TO_INDEX["T"] > 0)


    def check_shoot_the_moon(self, score_cards):
        point_players = set()
        for player_idx, cards in enumerate(score_cards):
            if cards.get(SUIT_TO_INDEX["S"], 0) & NUM_TO_INDEX["Q"]:
                point_players.add(player_idx)

            if cards.get(SUIT_TO_INDEX["H"], 0) > 0:
                point_players.add(player_idx)

        if len(point_players) > 1:
            self.is_shoot_the_moon = False


    def get_seen_cards(self):
        cards = copy.copy(EMPTY_CARDS)
        for trick in self.tricks:
            for suit, rank in trick[1]:
                cards[suit] |= rank

        return cards


    def get_remaining_cards(self):
        cards = copy.copy(FULL_CARDS)

        seen_cards = self.get_seen_cards()
        for suit, ranks in seen_cards.items():
            cards[suit] ^= ranks

        for suit, ranks in self.hand_cards.items():
            cards[suit] ^= ranks

        return cards


    def check_hearts_broken(self):
        has_point_players = set()

        for trick_idx, (winner_idx, trick) in enumerate(self.tricks):
            has_point_card = False
            for suit, rank in trick:
                if suit == SUIT_TO_INDEX["H"]:
                    has_point_players.add(winner_idx)
                elif suit == SUIT_TO_INDEX["S"] and rank & NUM_TO_INDEX["Q"]:
                    has_point_players.add(winner_idx)

            if trick_idx == len(self.tricks)-1:
                for suit, rank in trick:
                    self.is_hearts_broken = self.is_hearts_broken or (suit == SUIT_TO_INDEX["H"])
                    self.is_show_pig_card = self.is_show_pig_card or (suit == SUIT_TO_INDEX["S"] and rank & NUM_TO_INDEX["Q"] > 0)
                    self.is_show_double_card = self.is_show_double_card or (suit == SUIT_TO_INDEX["C"] and rank & NUM_TO_INDEX["T"] > 0)

        if len(has_point_players) > 1:
            self.is_shoot_the_moon = False


    def is_all_point_cards(self, cards):
        if cards.get(SUIT_TO_INDEX["D"], 0) == 0 and cards.get(SUIT_TO_INDEX["C"], 0) == 0:
            ranks = cards.get(SUIT_TO_INDEX["S"], 0)
            if ranks == 0:
                return True
            elif ranks == NUM_TO_INDEX["Q"]:
                return True
            else:
                return False
        else:
            return False


    def get_myself_valid_cards(self, hand_cards, current_round_idx=None):
        return self.get_valid_cards(copy.copy(hand_cards), current_round_idx)


    def get_valid_cards(self, cards, current_round_idx):
        if current_round_idx is None:
            current_round_idx = len(self.tricks)

        if current_round_idx == 1:
            if len(self.tricks[-1][1]) == 0:
                if cards.get(SUIT_TO_INDEX["C"], 0) & NUM_TO_INDEX["2"]:
                    return {SUIT_TO_INDEX["C"]: NUM_TO_INDEX["2"]}
            elif cards.get(SUIT_TO_INDEX["C"], 0) > 0:
                return {SUIT_TO_INDEX["C"]: cards[SUIT_TO_INDEX["C"]]}
            else:
                spades_rank = cards.get(SUIT_TO_INDEX["S"], 0)
                if spades_rank & NUM_TO_INDEX["Q"]:
                    spades_rank ^= NUM_TO_INDEX["Q"]

                return {SUIT_TO_INDEX["D"]: cards.get(SUIT_TO_INDEX["D"], 0), SUIT_TO_INDEX["S"]: spades_rank}
        else:
            if len(self.tricks[-1][1]) == 0:
                if self.is_hearts_broken or self.is_all_point_cards(cards):
                    return cards
                else:
                    if SUIT_TO_INDEX["H"] in cards: del cards[SUIT_TO_INDEX["H"]]
                    if cards.get(SUIT_TO_INDEX["S"], 0) & NUM_TO_INDEX["Q"]: cards[SUIT_TO_INDEX["S"]] ^= NUM_TO_INDEX["Q"]

                    return cards
            else:
                leading_suit = self.tricks[-1][1][0][0]

                if cards.get(leading_suit, 0) > 0:
                    return {leading_suit: cards[leading_suit]}
                else:
                    return cards


    def winning_index(self, trick):
        leading_suit, leading_rank = trick[1][0]

        winning_index = 0
        winning_card = trick[1][0]
        for i, (suit, rank) in enumerate(trick[1][1:]):
            if suit == leading_suit and rank > leading_rank:
                winning_index = i+1
                winning_card = [suit, rank]

                leading_rank = rank

        return winning_index, winning_card


    def remove_card(self, hand_cards, removed_card):
        hand_cards[removed_card[0]] ^= removed_card[1]


    def run(self, current_round_idx, played_card=None, selection_func=random_choose):
        first_choose = greedy_choose if IS_DEBUG else random_choose

        start_pos = None
        current_trick = self.tricks[-1]
        if current_trick:
            current_index = len(current_trick[1])

            if played_card is None:
                valid_cards = self.get_myself_valid_cards(self.hand_cards[self.position], current_round_idx)
                candicated_cards, played_card = first_choose(valid_cards, current_trick[1], self.is_hearts_broken, self.is_show_pig_card, self.is_show_double_card)

            self.played_card = played_card

            current_trick[1].append(played_card)
            self.remove_card(self.hand_cards[self.position], played_card)

            for trick_idx in range(4-len(current_trick[1])):
                player_idx = (self.position+(trick_idx+1))%4

                valid_cards = self.get_myself_valid_cards(self.hand_cards[player_idx], current_round_idx)
                #if sum(valid_cards.values()) == 0:
                #    print(1111, current_round_idx, player_idx, "--->", self.hand_cards, valid_cards)

                _, played_card = selection_func(valid_cards, current_trick[1], self.is_hearts_broken, self.is_show_pig_card, self.is_show_double_card)

                current_trick[1].append(played_card)
                self.remove_card(self.hand_cards[player_idx], played_card)

            winning_index, winning_card = self.winning_index(current_trick)
            start_pos = (self.position+(winning_index-current_index))%4

            current_trick[0] = start_pos
        else:
            start_pos = (self.position+1)%4

        for suit, rank in current_trick[1]:
            self.score_cards[start_pos][suit] |= rank

        if IS_DEBUG:
            print("trick_no={}, trick={}, is_hearts_broken={}, is_show_pig_card={}, is_show_double_card={}, is_shoot_the_moon={}, winning_card={}, start_pos={}".format(\
                current_round_idx, self.translate_trick_cards(), self.is_hearts_broken, self.is_show_pig_card, self.is_show_double_card, self.is_shoot_the_moon,\
                winning_card, start_pos))
            for player_idx in range(4):
                print("\tplayer-{}'s hand_cards is {}".format(player_idx, sorted(self.translate_hand_cards(self.hand_cards[player_idx]))))

        candicated_cards = [[], [], [], []]
        for round_idx in range(13-current_round_idx):
            self.tricks.append([None, []])

            current_index = None
            for trick_idx in range(4):
                if start_pos == self.position:
                    current_index = trick_idx

                valid_cards = self.get_myself_valid_cards(self.hand_cards[start_pos])
                #if sum(valid_cards.values()) == 0:
                #    print(2222, round_idx, current_round_idx, start_pos, "--->", self.hand_cards[start_pos], valid_cards)

                ccards, played_card = selection_func(valid_cards, self.tricks[-1][1], self.is_hearts_broken, self.is_show_pig_card, self.is_show_double_card)

                #print("start_pos={}, hand_cards={}, valid_cards={}, played_card={}".format(start_pos, self.hand_cards[start_pos], valid_cards, played_card))

                self.tricks[-1][1].append(played_card)
                self.remove_card(self.hand_cards[start_pos], played_card)

                start_pos = (start_pos+1)%4

            winning_index, winning_card = self.winning_index(self.tricks[-1])
            start_pos = (self.position+(winning_index-current_index))%4

            self.tricks[-1][0] = (start_pos+3)%4

            self.check_hearts_broken()

            for suit, rank in self.tricks[-1][1]:
                self.score_cards[start_pos][suit] |= rank

            if IS_DEBUG:
                print("trick_no={}, trick={}, is_hearts_broken={}, is_show_pig_card={}, is_show_double_card={}, is_shoot_the_moon={}, winning_card={}, start_pos={}".format(\
                    current_round_idx+round_idx+1, self.translate_trick_cards(), self.is_hearts_broken, self.is_show_pig_card, self.is_show_double_card, self.is_shoot_the_moon, \
                    winning_card, start_pos))

                for player_idx in range(4):
                    print("\tplayer-{}'s hand_cards is {}".format(player_idx, sorted(self.translate_hand_cards(self.hand_cards[player_idx]))))


    def score(self):
        scores = [count_points(cards, self.expose_hearts_ace) for player_idx, cards in enumerate(self.score_cards)]

        max_score = np.max(scores)
        if np.sum([1 for score in scores if score == 0]) == 3:
            for player_idx, score in enumerate(scores):
                scores[player_idx] = max_score if score == 0 else 0

        return scores


    def translate_trick_cards(self):
        cards = []
        for suit, rank in self.tricks[-1][1]:
            cards.append("{}{}".format(INDEX_TO_NUM[rank], INDEX_TO_SUIT[suit]))

        return cards


    def translate_hand_cards(self, hand_cards):
        cards = []

        for suit, ranks in hand_cards.items():
            bit_mask = 1
            while bit_mask <= 4096:
                if hand_cards[suit] & bit_mask:
                    cards.append(transform(INDEX_TO_NUM[bit_mask], INDEX_TO_SUIT[suit]))

                bit_mask <<= 1

        return cards


    def translate_taken_cards(self, hand_cards):
        cards = []

        bit_mask = 1
        while bit_mask <= 4096:
            if hand_cards[SUIT_TO_INDEX["H"]] & bit_mask:
                cards.append(transform(INDEX_TO_NUM[bit_mask], "H"))

            bit_mask <<= 1

        if hand_cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["Q"]:
            cards.append(Card(Suit.spades, Rank.queen))

        if hand_cards[SUIT_TO_INDEX["C"]] & NUM_TO_INDEX["T"]:
            cards.append(Card(Suit.clubs, Rank.ten))

        return cards


    def print_tricks(self, scores):
        for round_idx in range(1, len(self.tricks)+1):
            print("Round {:2d}: Tricks: {}".format(\
                round_idx, ["{}{}".format(INDEX_TO_NUM[rank], INDEX_TO_SUIT[suit]) for suit, rank in self.tricks[round_idx-1][1]]))

        for player_idx, score in enumerate(scores):
            print("player-{} get {} scores({}, expose_hearts_ace={})".format(player_idx, score, self.translate_taken_cards(self.score_cards[player_idx]), self.expose_hearts_ace))


def simulation(current_round_idx, position, hand_cards, tricks,
               score_cards=None, is_hearts_borken=False, expose_hearts_ace=False, played_card=None, selection_func=None):

    sm = None
    try:
        sm = SimpleGame(position=position, 
                        hand_cards=hand_cards, 
                        score_cards=score_cards, 
                        is_hearts_borken=is_hearts_borken, 
                        expose_hearts_ace=expose_hearts_ace, 
                        tricks=tricks)

        for player_idx, cards in enumerate(hand_cards):
            hand_cards[player_idx] = str_to_bitmask(cards)

            if IS_DEBUG:
                print("player-{}'s hand_cards is {}".format(player_idx, sm.translate_hand_cards(hand_cards[player_idx])))

        #valid_cards = sm.get_myself_valid_cards(sm.hand_cards[position], current_round_idx)

        sm.run(current_round_idx, played_card=played_card, selection_func=selection_func)
        scores = sm.score()

        if IS_DEBUG:
            sm.print_tricks(scores)
            print()

        return sm.played_card, scores, sm.score_cards
    except:
        for player_idx, cards in enumerate(hand_cards):
            print("player-{}'s hand_cards is {}".format(player_idx, sm.translate_hand_cards(hand_cards[player_idx])))
        print()

        raise

def run_simulation(seed,
                   current_round_idx, position, init_trick, hand_cards, is_hearts_broken, expose_hearts_ace, cards,
                   score_cards=None, played_card=None, selection_func=random_choose, must_have={}, void_info={}, simulation_time=0.93):
    stime = time.time()

    for trick_idx, (winner_index, trick) in enumerate(init_trick):
        for card_idx, card in enumerate(trick):
            for suit, rank in str_to_bitmask([card]).items():
                trick[card_idx] = [suit, rank]

    if played_card is not None:
        played_card = [SUIT_TO_INDEX[played_card.suit.__repr__()], NUM_TO_INDEX[played_card.rank.__repr__()]]

    simulations_cards = None
    if cards:
        simulation_cards = redistribute_cards(seed, position, hand_cards, init_trick[-1][1], cards, must_have, void_info)
    else:
        simulation_cards = [hand_cards]

    num = 0
    results = defaultdict(list)
    for simulation_card in simulation_cards:
        card, scores, _ = simulation(current_round_idx, 
                                     position, 
                                     simulation_card, 
                                     copy.deepcopy(init_trick), 
                                     copy.deepcopy(score_cards), 
                                     is_hearts_broken, 
                                     expose_hearts_ace, 
                                     played_card, 
                                     selection_func)

        min_score, other_score = None, 0
        for idx, score in enumerate(sorted(scores)):
            if idx == 0:
                min_score = score
            else:
                other_score += score

        self_score = scores[position]
        if self_score == min_score:
            self_score -= other_score/3
        else:
            sum_of_min_scores = [score for player_idx, score in enumerate(scores) if player_idx != position and score < self_score]
            self_score -= min_score

        results[tuple(card)].append(self_score)

        num += 1

        if time.time()-stime > simulation_time or IS_DEBUG:
            break

    return dict([(transform(INDEX_TO_NUM[card[1]], INDEX_TO_SUIT[card[0]]), scores) for card, scores in results.items()])


if __name__ == "__main__":
    import multiprocessing as mp

    position = 3
    expose_hearts_ace = False

    init_trick = [[None, [Card(Suit.clubs, Rank.two), Card(Suit.clubs, Rank.ace), Card(Suit.clubs, Rank.eight)]]]
    current_round_idx = 1

    hand_cards = [[transform(card[0], card[1]) for card in "KC,QC,TD,8H,9D,9S,5C,3D,3S,AH,QH,JH".split(",")],
                  [transform(card[0], card[1]) for card in "7H,8D,7C,5H,6C,6D,6S,5D,4D,2D,KS,AD".split(",")],
                  [transform(card[0], card[1]) for card in "TH,9H,TS,8S,6H,7D,4H,3C,2S,QS,AS,KH".split(",")],
                  [transform(card[0], card[1]) for card in "TC,JC,JS,9C,7S,5S,3H,4C,4S,2H,KD,QD,JD".split(",")]]

    init_trick = [[None, [Card(Suit.clubs, Rank.two), Card(Suit.clubs, Rank.three)]]]
    hand_cards = [[transform(card[0], card[1]) for card in "KC,JH,JC,TD,8S,6H,6C,5C,3H,2S,KD,TC,JD".split(",")],
                  [transform(card[0], card[1]) for card in "QC,JS,8H,7C,7D,5H,4D,4S,2D,KH,AC,QD".split(",")],
                  [transform(card[0], card[1]) for card in "9H,TS,9S,7H,6D,4H,5D,2H,3S,AS,KS,AH".split(",")],
                  [transform(card[0], card[1]) for card in "TH,9C,9D,8C,8D,7S,6S,5S,4C,3D,QS,AD,QH".split(",")]]

    myself_hand_cards = hand_cards

    cards = []

    played_card = None
    score_cards = None
    must_have = {}
    void_info = {}

    is_hearts_broken = False
    IS_DEBUG = True

    pool = mp.Pool(processes=1)

    selection_func = expert_choose
    mul_result = [pool.apply_async(run_simulation, args=(seed,
                                                         current_round_idx, 
                                                         position, 
                                                         init_trick, 
                                                         myself_hand_cards, 
                                                         is_hearts_broken, 
                                                         expose_hearts_ace, 
                                                         cards, 
                                                         score_cards, 
                                                         None, 
                                                         selection_func,
                                                         must_have, 
                                                         void_info,
                                                         0.935)) for seed in range(1)]

    results = defaultdict(list)
    partial_results = [res.get() for res in mul_result]
    for row in partial_results:
        for card, scores in row.items():
            results[card].extend(scores)

    pool.close()

    for played_card, scores in results.items():
        print(played_card, len(scores), np.mean(scores))
