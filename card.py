"""
This module contains the definition of types fundamental to card games,
most notably the type Card.
"""

import os
import sys
import glob

import time
import pickle

from random import shuffle

from orderedenum import OrderedEnum


class Suit(OrderedEnum):
    clubs = 0
    diamonds = 1
    spades = 2
    hearts = 3

    def __repr__(self):
        if sys.stdout.encoding in ['cp437', 'cp850']:
            # These are the correct unicode symbols in cp437 or cp850 encoding
            # They don't work for 1252 of utf8
            return [chr(5), chr(4), chr(6), chr(3)][self.value]
        else:
            return ["C", "D", "S", "H"][self.value]

    def __hash__(self):
        return hash(self.value)


class Rank(OrderedEnum):
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eight = 8
    nine = 9
    ten = 10
    jack = 11
    queen = 12
    king = 13
    ace = 14

    def __repr__(self):
        if self.value < 10:
            return str(self.value)
        else:
            return ['T', 'J', 'Q', 'K', 'A'][self.value - 10]

    def __hash__(self):
        return hash(self.value)


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return repr(self.rank) + repr(self.suit)

    def __lt__(self, other):
        return (self.suit, self.rank) < (other.suit, other.rank)

    def __eq__(self, other):
        try:
            is_equal = (self.suit, self.rank) == (other.suit, other.rank)

            return is_equal
        except:
            print(self, other)

            raise

    def __hash__(self):
        return hash(self.rank) + hash(self.suit)


SPADES_Q, SPADES_K, SPADES_A = Card(Suit.spades, Rank.queen), Card(Suit.spades, Rank.king), Card(Suit.spades, Rank.ace)
HEARTS_Q, HEARTS_K, HEARTS_A = Card(Suit.hearts, Rank.queen), Card(Suit.hearts, Rank.king), Card(Suit.hearts, Rank.ace)
CLUBS_T = Card(Suit.clubs, Rank.ten)


class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]


    def deal(self, cards=None):
        """
        Shuffles the cards and returns 4 lists of 13 cards.
        """

        cards = cards if cards is not None else self.cards

        shuffle(self.cards)
        for i in range(0, 52, 13):
            yield sorted(self.cards[i:i + 13])


def transform(rank, suit):
    if suit == "S":
        suit = Suit.spades
    elif suit == "H":
        suit = Suit.hearts
    elif suit == "C":
        suit = Suit.clubs
    elif suit == "D":
        suit = Suit.diamonds

    if rank == "A":
        rank = Rank.ace
    elif rank == "2":
        rank = Rank.two
    elif rank == "3":
        rank = Rank.three
    elif rank == "4":
        rank = Rank.four
    elif rank == "5":
        rank = Rank.five
    elif rank == "6":
        rank = Rank.six
    elif rank == "7":
        rank = Rank.seven
    elif rank == "8":
        rank = Rank.eight
    elif rank == "9":
        rank = Rank.nine
    elif rank == "10" or rank == "T":
        rank = Rank.ten
    elif rank == "J":
        rank = Rank.jack
    elif rank == "Q":
        rank = Rank.queen
    elif rank == "K":
        rank = Rank.king

    return Card(suit, rank)


def generate_card_games(num_of_game, filepath_out):
    deck = Deck()

    cards = []
    with open(filepath_out, "wb") as out_file:
        for _ in range(num_of_game):
            cards.append(list(deck.deal()))

        pickle.dump(cards, out_file)


def read_card_games(folder):
    cards = []

    for filepath_in in glob.glob(folder):
        print("start to read {}".format(filepath_in))
        with open(filepath_in, "rb") as in_file:
            cards.extend(pickle.load(in_file))

    print("We read {} pre-setting cards".format(len(cards)))

    return cards


POINT_CARDS = {Card(Suit.clubs, Rank.ten), Card(Suit.spades, Rank.queen), Card(Suit.hearts, Rank.ace),
               Card(Suit.hearts, Rank.two), Card(Suit.hearts, Rank.three), Card(Suit.hearts, Rank.four),
               Card(Suit.hearts, Rank.five), Card(Suit.hearts, Rank.six), Card(Suit.hearts, Rank.seven),
               Card(Suit.hearts, Rank.eight), Card(Suit.hearts, Rank.nine), Card(Suit.hearts, Rank.ten),
               Card(Suit.hearts, Rank.jack), Card(Suit.hearts, Rank.queen), Card(Suit.hearts, Rank.king)}


NUM_TO_INDEX = {"2":1, "3":2, "4":4, "5":8, "6":16, "7":32, "8":64, "9":128, "T":256, "J":512, "Q":1024, "K":2048, "A":4096}
INDEX_TO_NUM = {1:"2", 2:"3", 4:"4", 8:"5", 16:"6", 32:"7", 64:"8", 128:"9", 256:"T", 512:"J", 1024:"Q", 2048:"K", 4096:"A"}

SUIT_TO_INDEX = {"C":0, "D":1, "S":2, "H":3}
INDEX_TO_SUIT = {0:"C", 1:"D", 2:"S", 3:"H"}

RANK_SUM = sum([2**idx for idx in range(0, 13)])
EMPTY_CARDS = {SUIT_TO_INDEX["C"]: 0, SUIT_TO_INDEX["D"]: 0, SUIT_TO_INDEX["S"]: 0, SUIT_TO_INDEX["H"]: 0}
FULL_CARDS = {SUIT_TO_INDEX["C"]: RANK_SUM, SUIT_TO_INDEX["D"]: RANK_SUM, SUIT_TO_INDEX["S"]: RANK_SUM, SUIT_TO_INDEX["H"]: RANK_SUM}


def _card_to_index(self, card_string):
    global NUM_TO_INDEX, SUIT_TO_INDEX

    return NUM_TO_INDEX[card_string[0]], SUIT_TO_INDEX[card_string[1]]


def _index_to_card(self, n, s):
    global INDEX_TO_NUM, INDEX_TO_SUIT

    return INDEX_TO_NUM[n] + INDEX_TO_SUIT[s]


def str_to_bitmask(cards):
    global NUM_TO_INDEX, SUIT_TO_INDEX

    hand_cards = {}
    for card in cards:
        card = str(card)
        rank, suit = card[0], card[1]
        index = SUIT_TO_INDEX[suit]

        hand_cards.setdefault(index, 0)
        hand_cards[index] |= NUM_TO_INDEX[rank]

    return hand_cards


def bitmask_to_str(suit, rank):
    return "{}{}".format(INDEX_TO_NUM[rank], INDEX_TO_SUIT[suit])


def card_to_bitmask(cards):
    global NUM_TO_INDEX, SUIT_TO_INDEX

    hand_cards = [0, 0, 0, 0]
    for card in cards:
        rank, suit = 1 << (card.rank.value-2), card.suit.value

        hand_cards[suit] |= rank

    return hand_cards


def batch_bitmask_to_card(suit, ranks):
    bitmask = NUM_TO_INDEX["2"]

    while bitmask <= NUM_TO_INDEX["A"]:
        if ranks & bitmask:
            yield bitmask_to_card(suit, bitmask)

        bitmask <<= 1


def bitmask_to_card(suit, rank):
    return transform(INDEX_TO_NUM[rank], INDEX_TO_SUIT[suit])


def count_points(cards, expose_hearts_ace):
    global SUIT_TO_INDEX, NUM_TO_INDEX

    point = 0
    bit_mask = NUM_TO_INDEX["2"]
    while bit_mask <= NUM_TO_INDEX["A"]:
        if cards[SUIT_TO_INDEX["H"]] & bit_mask:
            point += 1*expose_hearts_ace

        bit_mask <<= 1

    if cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["Q"]:
        point += 13

    if cards[SUIT_TO_INDEX["C"]] & NUM_TO_INDEX["T"]:
        point <<= 1

    return point


def translate_hand_cards(hand_cards, is_bitmask=False):
    cards = []

    for suit, ranks in hand_cards.items():
        bit_mask = 1
        while bit_mask <= NUM_TO_INDEX["A"]:
            if hand_cards[suit] & bit_mask:
                if is_bitmask:
                    cards.append((suit, bit_mask))
                else:
                    cards.append(transform(INDEX_TO_NUM[bit_mask], INDEX_TO_SUIT[suit]))

            bit_mask <<= 1

    return cards


def get_remaining_cards(trick_nr, init_trick, score_cards):
    played_cards = {}
    for cards in score_cards:
        for suit, ranks in enumerate(cards):
            played_cards.setdefault(suit, 0)
            played_cards[suit] |= ranks

    for suit, rank in init_trick[-1][1]:
        played_cards[suit] |= rank

    if trick_nr == 1:
        played_cards[SUIT_TO_INDEX["H"]] = 8191
        played_cards[SUIT_TO_INDEX["S"]] |= NUM_TO_INDEX["Q"]

    probs, size = [], 0
    for suit, ranks in FULL_CARDS.items():
        bit_mask = NUM_TO_INDEX["2"]

        while bit_mask <= NUM_TO_INDEX["A"]:
            prob = 0.0
            if played_cards.get(suit, 0) & bit_mask == 0:
                prob = 1.0

                size += 1

            probs.append([(suit, bit_mask), prob])

            bit_mask <<= 1

    results = []
    if trick_nr != 1 or (trick_nr == 1 and len(init_trick[-1][1]) > 0):
        results = [[card, prob/size] for card, prob in probs]

    return results


if __name__ == "__main__":
    num_of_games, sub_folder = int(sys.argv[1]), int(sys.argv[2])
    filepath_in = "game/game_{:04d}/{:02d}/game_{}.pkl".format(\
        num_of_games, sub_folder, int(time.time()))

    if not os.path.exists(os.path.dirname(filepath_in)):
        os.makedirs(os.path.dirname(filepath_in))

    generate_card_games(int(sys.argv[1]), filepath_in)

    #for card in read_card_games(filepath_in):
    #    print(card)
