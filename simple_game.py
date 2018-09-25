#`!/usr/bin/env python

import time
import copy

import numpy as np

from collections import defaultdict

from card import Rank, Suit, Card, Deck

from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import EMPTY_CARDS, FULL_CARDS
from card import str_to_bitmask, count_points

from rules import is_card_valid, transform
from redistribute_cards import redistribute_cards


def random_choose(cards, trick):
    candicated_cards = []

    for suit, ranks in cards.items():
        bit_mask = 1
        while bit_mask <= 4096:
            if ranks & bit_mask:
                candicated_cards.append([suit, bit_mask])

            bit_mask <<= 1

    return candicated_cards, candicated_cards[np.random.choice(len(candicated_cards))]


def greedy_choose(cards, trick, is_hearts_broken=False, is_pig_card_taken=False):
    def choose_max_card(cards, suits=[SUIT_TO_INDEX["S"], SUIT_TO_INDEX["H"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["C"]]):
        candicated_cards, played_card = [], None

        bit_mask = 4096
        while bit_mask > 0:
            for suit in suits:
                rank = cards.get(suit, 0)
                if rank & bit_mask:
                    if played_card is None:
                        played_card = [suit, bit_mask]

                        return candicated_cards, played_card

                    candicated_cards.append([suit, bit_mask])

            bit_mask >>= 1

        return candicated_cards, played_card


    def choose_min_card(cards, suits=[SUIT_TO_INDEX["S"], SUIT_TO_INDEX["H"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["C"]]):
        candicated_cards, played_card = [], None

        bit_mask = 1
        while bit_mask <= 4096:
            for suit in suits:
                rank = cards.get(suit, 0)
                if rank & bit_mask:
                    if played_card is None:
                        played_card = [suit, bit_mask]

                        return candicated_cards, played_card

                    candicated_cards.append([suit, bit_mask])

            bit_mask <<= 1

        return candicated_cards, played_card


    safe_play, candicated_cards = None, []
    if trick:
        leading_suit, max_rank = trick[0][0], trick[0][1]

        for suit, rank in trick[1:]:
            if suit == leading_suit and rank > max_rank:
                max_rank = rank

        eaten_play = None
        rank = cards.get(leading_suit, 0)

        if rank > 0:
            bit_mask = 4096
            while bit_mask > 0:
                if rank & bit_mask:
                    if bit_mask < max_rank:
                        if safe_play is None: safe_play = [leading_suit, bit_mask]
                    else:
                        if eaten_play is None: eaten_play = [leading_suit, bit_mask]

                    if safe_play is not None and eaten_play is not None:
                        break

                    candicated_cards.append([leading_suit, bit_mask])

                bit_mask >>= 1

            if safe_play is None:
                safe_play = eaten_play
        else:
            if cards.get(SUIT_TO_INDEX["S"], 0) & NUM_TO_INDEX["Q"]:
                safe_play = [SUIT_TO_INDEX["S"], NUM_TO_INDEX["Q"]]
            elif is_pig_card_taken == False and cards.get(SUIT_TO_INDEX["S"], 0) >= 2048:
                safe_play = [SUIT_TO_INDEX["S"], NUM_TO_INDEX["A"]]  if cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["A"] else [SUIT_TO_INDEX["S"], NUM_TO_INDEX["K"]]
            elif cards.get(SUIT_TO_INDEX["H"], 0) >= 128:
                candicated_cards, safe_play = choose_max_card(cards, suits=[SUIT_TO_INDEX["H"]])
            else:
                candicated_cards, safe_play = choose_max_card(cards)
    else:
        candicated_cards, safe_play = choose_min_card(cards)

    return candicated_cards, safe_play


class SimpleGame(object):
    def __init__(self,
                 position,
                 hand_cards,
                 score_cards=None,
                 is_hearts_borken=False,
                 expose_hearts_ace=False,
                 tricks=[]):

        self.position = position
        self.hand_cards = hand_cards

        self.is_pig_card_taken = False

        if score_cards is None:
            self.score_cards = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        else:
            self.score_cards = score_cards
            for cards in self.score_cards:
                if self.is_pig_card_taken:
                    break
                else:
                    self.is_take_pig_card(cards)

        #print("current score_cards: ", self.score_cards)

        self.tricks = tricks

        self.is_hearts_broken = is_hearts_borken
        self.expose_hearts_ace = 2 if expose_hearts_ace else 1

        self.played_card = None


    def is_take_pig_card(self, cards):
        if self.is_pig_card_taken == False:
            if cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["Q"]:
                self.is_pig_card_taken = True


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
        if self.is_hearts_broken == False or self.is_pig_card_taken == False:
            trick = self.tricks[-1][1]
            for suit, rank in trick:
               if suit == SUIT_TO_INDEX["H"]:
                    self.is_hearts_broken = True

               if suit == SUIT_TO_INDEX["S"] and rank & NUM_TO_INDEX["Q"]:
                   self.is_pig_card_taken = True


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
        start_pos = None
        current_trick = self.tricks[-1]
        if current_trick:
            current_index = len(current_trick[1])

            if played_card is None:
                valid_cards = self.get_myself_valid_cards(self.hand_cards[self.position], current_round_idx)
                candicated_cards, played_card = random_choose(valid_cards, current_trick[1])

            self.played_card = played_card

            current_trick[1].append(played_card)
            self.remove_card(self.hand_cards[self.position], played_card)

            for trick_idx in range(4-len(current_trick[1])):
                player_idx = (self.position+(trick_idx+1))%4

                valid_cards = self.get_myself_valid_cards(self.hand_cards[player_idx], current_round_idx)
                if sum(valid_cards.values()) == 0:
                    print(1111, current_round_idx, player_idx, "--->", self.hand_cards, valid_cards)

                _, played_card = selection_func(valid_cards, current_trick[1])

                current_trick[1].append(played_card)
                self.remove_card(self.hand_cards[player_idx], played_card)

            winning_index, winning_card = self.winning_index(current_trick)
            start_pos = (self.position+(winning_index-current_index))%4

            current_trick[0] = start_pos
        else:
            start_pos = (self.position+1)%4

        for suit, rank in current_trick[1]:
            self.score_cards[start_pos][suit] |= rank

        """
        print("trick_no, trick, is_hearts_broken, winning_index, winning_card, current_index, start_pos = ({}, {}, {}, {}, {}, {}, {})".format(\
              current_round_idx, self.translate_trick_cards(), self.is_hearts_broken, winning_index, winning_card, current_index, start_pos))
        for player_idx in range(4):
            print("\tplayer-{}'s hand_cards is {}".format(player_idx, sorted(self.translate_hand_cards(self.hand_cards[player_idx]))))
        """

        candicated_cards = [[], [], [], []]
        for round_idx in range(13-current_round_idx):
            self.tricks.append([None, []])

            current_index = None
            for trick_idx in range(4):
                if start_pos == self.position:
                    current_index = trick_idx

                valid_cards = self.get_myself_valid_cards(self.hand_cards[start_pos])
                if sum(valid_cards.values()) == 0:
                    print(2222, round_idx, current_round_idx, start_pos, "--->", self.hand_cards[start_pos], valid_cards)
                ccards, played_card = selection_func(valid_cards, self.tricks[-1][1])

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

            """
            print("trick_no, trick, is_hearts_broken, winning_index, winning_card, current_index, start_pos = ({}, {}, {}, {}, {}, {}, {})".format(\
                current_round_idx+round_idx+1, self.translate_trick_cards(), self.is_hearts_broken, winning_index, winning_card, current_index, start_pos))

            for player_idx in range(4):
                print("\tplayer-{}'s hand_cards is {}".format(player_idx, sorted(self.translate_hand_cards(self.hand_cards[player_idx]))))
            """


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
            #print("player-{}'s hand_cards is {}".format(player_idx, sm.translate_hand_cards(hand_cards[player_idx])))

        sm.run(current_round_idx, played_card=played_card, selection_func=greedy_choose)
        scores = sm.score()

        #sm.print_tricks(scores)

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

    simulation_cards = redistribute_cards(seed, position, hand_cards, init_trick[-1][1], cards, must_have, void_info)

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
            self_score -= min_score

        results[tuple(card)].append(self_score)

        num += 1

        if time.time()-stime > simulation_time:
            break

    return dict([(transform(INDEX_TO_NUM[card[1]], INDEX_TO_SUIT[card[0]]), scores) for card, scores in results.items()])


if __name__ == "__main__":
    import multiprocessing as mp

    position = 3

    """
    init_trick = [[2, [Card(Suit.clubs, Rank.two), Card(Suit.clubs, Rank.five), Card(Suit.clubs, Rank.ace), Card(Suit.clubs, Rank.three)]],
                  [None, [Card(Suit.spades, Rank.two)]]]
    current_round_idx = len(init_trick)

    myself_hand_cards = [transform(card[0], card[1]) for card in "TS,4S,5H,6H,KC,9C,4C,6C,8D,7D,4D,2D".split(",")]

    cards = Deck().cards
    for player_idx, cs in init_trick:
        for card in cs:
            cards.remove(card)

    for card in myself_hand_cards:
        cards.remove(card)

    myself_hand_cards = [[] if player_idx != position else myself_hand_cards for player_idx in range(4)]

    played_card = None
    score_cards = None

    must_have = {0: [transform("T", "H"), transform("Q", "S"), transform("K", "S")]}
    void_info = {0: {Suit.clubs: False, Suit.hearts: False, Suit.spades: False, Suit.diamonds: True},
                 1: {Suit.clubs: False, Suit.hearts: True, Suit.spades: False, Suit.diamonds: True},
                 2: {Suit.clubs: False, Suit.hearts: False, Suit.spades: True, Suit.diamonds: False}}
    """

    is_hearts_broken = True

    init_trick = [[None, [Card(Suit.hearts, Rank.five)]]]
    current_round_idx = 10

    myself_hand_cards = [transform(card[0], card[1]) for card in "4D,5D,6S,AS".split(",")]
    played_card = None
    score_cards = None

    must_have = {1: [transform("Q", "S"), transform("Q", "H"), transform("J", "H")]}
    void_info = {0: {Suit.clubs: True, Suit.hearts: False, Suit.spades: False, Suit.diamonds: True},
                 1: {Suit.clubs: False, Suit.hearts: False, Suit.spades: True, Suit.diamonds: False},
                 2: {Suit.clubs: False, Suit.hearts: False, Suit.spades: True, Suit.diamonds: True}}

    is_hearts_broken = True

    myself_hand_cards = [[] if player_idx != position else myself_hand_cards for player_idx in range(4)]

    cards = [transform(card[0], card[1]) for card in "5C,8C,QC,QH,7S,2H,3H,4H,7D,8D,3S".split(",")]

    pool = mp.Pool(processes=1)

    selection_func = greedy_choose
    mul_result = [pool.apply_async(run_simulation, args=(seed,
                                                         current_round_idx, 
                                                         position, 
                                                         init_trick, 
                                                         myself_hand_cards, 
                                                         is_hearts_broken, 
                                                         False, 
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
