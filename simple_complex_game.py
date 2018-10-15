#!/usr/bin/env python

import sys

import time
import copy

import numpy as np

from collections import defaultdict

from card import Rank, Suit, Card, Deck

from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import EMPTY_CARDS, FULL_CARDS, RANK_SUM
from card import card_to_bitmask, str_to_bitmask, bitmask_to_str, count_points, translate_hand_cards

from rules import is_card_valid, transform
from redistribute_cards import redistribute_cards
from strategy_play import random_choose, greedy_choose
from expert_play import expert_choose


IS_DEBUG = False

ALL_SUIT = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["H"], SUIT_TO_INDEX["S"]]

class SimpleGame(object):
    def __init__(self,
                 position,
                 hand_cards,
                 void_info=None,
                 score_cards=None,
                 is_hearts_broken=False,
                 is_show_pig_card=False,
                 is_show_double_card=False,
                 has_point_players=set(),
                 expose_hearts_ace=False,
                 tricks=[]):

        self.position = position
        self.hand_cards = hand_cards

        self.current_info = [[13, RANK_SUM] for _ in ALL_SUIT]

        self.is_show_pig_card = is_show_pig_card
        self.is_show_double_card = is_show_double_card
        self.has_point_players = has_point_players

        self.void_info = dict([[player_idx, {SUIT_TO_INDEX["C"]: False, SUIT_TO_INDEX["D"]: False, SUIT_TO_INDEX["H"]: False, SUIT_TO_INDEX["S"]: False}] for player_idx in range(4)])
        if void_info:
            for player_idx, info in void_info.items():
                if player_idx in self.void_info:
                    for suit, is_void in info.items():
                        self.void_info[player_idx][SUIT_TO_INDEX[suit.__repr__()]] = is_void

        if score_cards is None:
            self.score_cards = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        else:
            self.score_cards = score_cards

            if not self.is_show_pig_card or not self.is_show_double_card:
                for cards in self.score_cards:
                    if not self.is_show_pig_card or not self.is_show_double_card:
                        self.check_show_pig_card(cards)
                        self.check_show_double_card(cards)

                    self.handle_current_info(cards)

            self.check_shoot_the_moon(self.score_cards)

        self.tricks = tricks

        self.played_card = None

        self.start_pos = self.position
        self.is_finished = False

        self.is_hearts_broken = is_hearts_broken
        self.expose_hearts_ace = 2 if expose_hearts_ace else 1


    def handle_current_info(self, cards):
        bit_mask = NUM_TO_INDEX["2"]
        while bit_mask <= NUM_TO_INDEX["A"]:
            for suit in ALL_SUIT:
                rank = None

                if cards[suit] & bit_mask:
                    self.current_info[suit][0] -= 1
                    self.current_info[suit][1] ^= bit_mask

            bit_mask <<= 1


    def check_show_pig_card(self, cards):
        self.is_show_pig_card = (cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["Q"] > 0)


    def check_show_double_card(self, cards):
        self.is_show_double_card = (cards[SUIT_TO_INDEX["C"]] & NUM_TO_INDEX["T"] > 0)


    def check_shoot_the_moon(self, score_cards):
        for player_idx, cards in enumerate(score_cards):
            if cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["Q"]:
                self.has_point_players.add(player_idx)

            if cards[SUIT_TO_INDEX["H"]] > 0:
                self.has_point_players.add(player_idx)


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


    def get_myself_valid_cards(self, hand_cards, current_round_idx):
        return self.get_valid_cards(copy.copy(hand_cards), current_round_idx)


    def get_valid_cards(self, cards, current_round_idx):
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


    def remove_card(self, hand_cards, removed_card):
        hand_cards[removed_card[0]] ^= removed_card[1]


    def add_card_to_trick(self, player_idx, card):
        self.tricks[-1][1].append(card)

        suit, rank = card
        self.current_info[suit][0] -= 1
        self.current_info[suit][1] ^= rank

        if self.tricks[-1][1][-1][0] != self.tricks[-1][1][0][0]:
            self.void_info[player_idx][self.tricks[-1][1][0][0]] = True


    def step(self, current_round_idx, selection_func, played_card=None):
        current_round_idx += len(self.tricks)-1

        def winning_index(trick):
            leading_suit, leading_rank = trick[1][0]

            winning_index = 0
            winning_card = trick[1][0]
            for i, (suit, rank) in enumerate(trick[1][1:]):
                if suit == leading_suit and rank > leading_rank:
                    winning_index = i+1
                    winning_card = [suit, rank]

                    leading_rank = rank

            return winning_index, winning_card

        if self.start_pos == self.position:
            self.current_index = len(self.tricks[-1][1])

        if played_card is None:
            valid_cards = self.get_myself_valid_cards(self.hand_cards[self.start_pos], current_round_idx)

            func = random_choose if self.start_pos == self.position and self.played_card is None else selection_func[self.start_pos]
            ccards, played_card = func(self.start_pos, 
                                       valid_cards, 
                                       self.tricks[-1][1], 
                                       self.hand_cards[self.start_pos].get(SUIT_TO_INDEX["S"], 0) & NUM_TO_INDEX["Q"] > 0,
                                       self.is_hearts_broken, 
                                       self.is_show_pig_card, 
                                       self.is_show_double_card, 
                                       self.has_point_players, 
                                       self.current_info, 
                                       self.void_info)

        #print(len(self.tricks), self.tricks[-1][1], self.hand_cards[self.start_pos], self.is_hearts_broken)
        if played_card is None:
            raise Exception("Impossible to get 'None' of played_card")

        if self.played_card is None:
            self.played_card = played_card

        self.add_card_to_trick(self.start_pos, played_card)
        self.remove_card(self.hand_cards[self.start_pos], played_card)

        if len(self.tricks[-1][1]) == 4:
            winning_index, winning_card = winning_index(self.tricks[-1])
            self.start_pos = (self.position+(winning_index-self.current_index))%4

            self.tricks[-1][0] = (self.start_pos)%4

            self.post_round_end()

            if IS_DEBUG:
                self.translate_current_info(current_round_idx, winning_card)
        else:
            self.start_pos = (self.start_pos+1)%4


    def run(self, current_round_idx, played_card=None, selection_func=random_choose):
        while any([rank > 0 for player_idx in range(4) for suit, rank in self.hand_cards[player_idx].items()]):
            self.step(current_round_idx, selection_func)


    def post_round_end(self):
        winner_idx = self.tricks[-1][0]
        for suit, rank in self.tricks[-1][1]:
            self.is_hearts_broken = self.is_hearts_broken or (suit == SUIT_TO_INDEX["H"])
            self.is_show_pig_card = self.is_show_pig_card or (suit == SUIT_TO_INDEX["S"] and rank & NUM_TO_INDEX["Q"] > 0)
            self.is_show_double_card = self.is_show_double_card or (suit == SUIT_TO_INDEX["C"] and rank & NUM_TO_INDEX["T"] > 0)

            if suit == SUIT_TO_INDEX["H"]:
                self.has_point_players.add(winner_idx)
            elif suit == SUIT_TO_INDEX["S"] and rank & NUM_TO_INDEX["Q"]:
                self.has_point_players.add(winner_idx)

            self.score_cards[self.start_pos][suit] |= rank

        if all([rank == 0 for player_idx in range(4) for suit, rank in self.hand_cards[player_idx].items()]):
            self.is_finished = True
        else:
            if len(self.tricks[-1][1]) == 4:
                self.tricks.append([None, []])


    def score(self):
        scores = [count_points(cards, self.expose_hearts_ace) for player_idx, cards in enumerate(self.score_cards)]

        is_my_shoot_the_moon = False
        max_score = np.max(scores)
        if np.sum([1 for score in scores if score == 0]) == 3:
            for player_idx, score in enumerate(scores):
                scores[player_idx] = max_score if score == 0 else 0

            if scores[self.position] == 0:
                is_my_shoot_the_moon = True

        return scores, is_my_shoot_the_moon


    def translate_current_info(self, current_round_idx, winning_card=None):
        def translate_trick_cards(trick):
            cards = []
            for suit, rank in trick:
                cards.append("{}{}".format(INDEX_TO_NUM[rank], INDEX_TO_SUIT[suit]))

            return cards

        if winning_card is not None:
            print("trick_no={}, trick={}, is_hearts_broken={}, is_show_pig_card={}, is_show_double_card={}, has_point_players={}, winning_card={}, start_pos={}".format(\
               current_round_idx, translate_trick_cards(self.tricks[-2][1]), self.is_hearts_broken, self.is_show_pig_card, self.is_show_double_card, self.has_point_players, \
               winning_card, self.start_pos))

        for suit in ALL_SUIT:
            cards = []

            bit_mask = NUM_TO_INDEX["2"]
            while bit_mask <= NUM_TO_INDEX["A"]:
                if self.current_info[suit][1] & bit_mask:
                    cards.append("{}{}".format(INDEX_TO_NUM[bit_mask], INDEX_TO_SUIT[suit]))

                bit_mask <<= 1

            print("suit: {}, remaining_cards in deck: {}({})".format(\
                INDEX_TO_SUIT[suit], cards, self.current_info[suit][0]))

        for player_idx in range(4):
          print("\tplayer-{}'s hand_cards is {}".format(player_idx, sorted(translate_hand_cards(self.hand_cards[player_idx]))))


    def translate_taken_cards(self, hand_cards):
        cards = []

        bit_mask = 1
        while bit_mask <= NUM_TO_INDEX["A"]:
            if hand_cards[SUIT_TO_INDEX["H"]] & bit_mask:
                cards.append(transform(INDEX_TO_NUM[bit_mask], "H"))

            bit_mask <<= 1

        if hand_cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["Q"]:
            cards.append(Card(Suit.spades, Rank.queen))

        if hand_cards[SUIT_TO_INDEX["C"]] & NUM_TO_INDEX["T"]:
            cards.append(Card(Suit.clubs, Rank.ten))

        return cards


    def print_tricks(self, scores):
        for round_idx in range(1, min(14, len(self.tricks)+1)):
            print("Round {:2d}: Tricks: {}".format(\
                round_idx, ["{}{}".format(INDEX_TO_NUM[rank], INDEX_TO_SUIT[suit]) for suit, rank in self.tricks[round_idx-1][1]]))

        for player_idx, score in enumerate(scores):
            print("player-{} get {:3d} scores({}, expose_hearts_ace={})".format(\
                player_idx, score, self.translate_taken_cards(self.score_cards[player_idx]), self.expose_hearts_ace))


def simulation(current_round_idx, position, hand_cards, tricks,
               void_info={}, score_cards=None, is_hearts_broken=False, expose_hearts_ace=False, played_card=None, selection_func=None,
               proactive_mode=None):

    sm = None
    try:
        for player_idx, cards in enumerate(hand_cards):
            hand_cards[player_idx] = str_to_bitmask(cards)

        sm = SimpleGame(position=position, 
                        hand_cards=hand_cards, 
                        void_info=void_info,
                        score_cards=score_cards, 
                        is_hearts_broken=is_hearts_broken, 
                        expose_hearts_ace=expose_hearts_ace, 
                        tricks=tricks)

        if IS_DEBUG:
            for player_idx, cards in enumerate(hand_cards):
                print("player-{}'s hand_cards is {}".format(player_idx, translate_hand_cards(hand_cards[player_idx])))

        if proactive_mode:
            ff = dict([[player_idx, random_choose if player_idx == position else np.random.choice(selection_func)] for player_idx in range(4)])
        elif len(selection_func) == 1:
            ff = dict([[player_idx, np.random.choice(selection_func)] for player_idx in range(4)])
        else:
            ff = dict(zip(range(4),  np.random.choice(selection_func, size=4, p=[0.5, 0.5])))

        sm.run(current_round_idx, played_card=played_card, selection_func=ff)
        scores, num_of_shoot_the_moon = sm.score()

        if IS_DEBUG:
            sm.print_tricks(scores)
            print()

        return sm.played_card, scores, sm.score_cards, num_of_shoot_the_moon
    except Exception as e:
        raise

        for player_idx, cards in enumerate(hand_cards):
            print("player-{}'s hand_cards is {}".format(player_idx, translate_hand_cards(hand_cards[player_idx])))
        print()

        return None, None, None, None


def run_simulation(seed, current_round_idx, position, init_trick, hand_cards, is_hearts_broken, expose_hearts_ace, cards,
                   score_cards=None, played_card=None, selection_func=random_choose, must_have={}, void_info={}, 
                   proactive_mode=None, simulation_time=0.93):

    simulation_time = max(simulation_time, 0.1)

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

    results, num_of_shoot_the_moon = defaultdict(list), defaultdict(int)
    for simulation_card in simulation_cards:
        card, scores, _, self_shoot_the_moon = simulation(current_round_idx, 
                                                          position, 
                                                          simulation_card, 
                                                          copy.deepcopy(init_trick), 
                                                          void_info,
                                                          copy.deepcopy(score_cards), 
                                                          is_hearts_broken, 
                                                          expose_hearts_ace, 
                                                          played_card, 
                                                          selection_func,
                                                          proactive_mode)

        if card is None:
            continue

        card = tuple(card)

        rating = [0, 0, 0, 0]

        info = zip(range(4), scores)
        pre_score, pre_rating, sum_score = None, None, np.array(scores)/np.sum(scores)
        for rating_idx, (player_idx, score) in enumerate(sorted(info, key=lambda x: -x[1])):
            tmp_rating = rating_idx
            if pre_score is not None:
                if score == pre_score:
                    tmp_rating = pre_rating

            rating[player_idx] = (4-tmp_rating)/4 + sum_score[player_idx]

            pre_score = score
            pre_rating = tmp_rating

        results[card].append([rating[position], self_shoot_the_moon])

        if time.time()-stime > simulation_time or IS_DEBUG:
            break

    return dict([(transform(INDEX_TO_NUM[card[1]], INDEX_TO_SUIT[card[0]]), info) for card, info in results.items()])


if __name__ == "__main__":
    import multiprocessing as mp

    position = 3
    current_round_idx = 1
    expose_hearts_ace = False
    is_hearts_broken = False

    """
    init_trick = [[None, [Card(Suit.clubs, Rank.two)]]]
    hand_1 = "".replace(" ", "")
    hand_2 = "".replace(" ", "")
    hand_3 = "".replace(" ", "")
    hand_4 = "".replace(" ", "")
    """

    init_trick = [[None, []]]
    hand_1 = "JH, TC, 7H, 8C, 6H, 5H, 6C, 3H, 2H, 3S, QS, QC, 9H".replace(" ", "")
    hand_2 = "TH, JD, 8H, 9S, 8S, 5C, 5S, 4D, 4S, 2S, AS, KC, QD".replace(" ", "")
    hand_3 = "AH, 9D, 7C, 7D, 6S, 4H, 5D, 4C, 3D, 2D, AD, QH, KD".replace(" ", "")
    hand_4 = "JC, JS, TD, TS, 9C, 8D, 7S, 6D, 3C, 2C, KS, KH, AC".replace(" ", "")

    #init_trick = [[None, [Card(Suit.clubs, Rank.two)]]]
    #hand_1 = "JH, TC, 7H, 8C, 6H, 5H, 6C, 3H, 2H, 3S, QS, QC, 9H".replace(" ", "")
    #hand_2 = "TH, JD, 8H, 9S, 8S, 5C, 5S, 4D, 4S, 2S, AS, KC, QD".replace(" ", "")
    #hand_3 = "AH, 9D, 7D, 6S, 4H, 5D, 4C, 3D, 2D, AD, QH, KD".replace(" ", "")
    #hand_4 = "JC, JS, TD, TS, 9C, 8D, 7S, 6D, 3C, 7C, KS, KH, AC".replace(" ", "")

    hand_cards = [[transform(card[0], card[1]) for card in hand_1.split(",")],
                  [transform(card[0], card[1]) for card in hand_2.split(",")],
                  [transform(card[0], card[1]) for card in hand_3.split(",")],
                  [transform(card[0], card[1]) for card in hand_4.split(",")]]

    myself_hand_cards = hand_cards

    cards = []

    played_card = None
    score_cards = None
    must_have = {}
    void_info = {}
    proactive_mode = None

    IS_DEBUG = True

    pool = mp.Pool(processes=1)

    selection_func = [expert_choose]
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
                                                         proactive_mode,
                                                         0.935)) for seed in range(1)]

    results = defaultdict(list)
    partial_results = [res.get() for res in mul_result]
    for row in partial_results:
        for card, scores in row.items():
            results[card].extend(scores)

    pool.close()

    for played_card, scores in results.items():
        print(played_card, len(scores), np.mean(scores))
