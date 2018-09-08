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



if __name__ == "__main__":
    num_of_games = int(sys.argv[1])
    filepath_in = "game/game_{:04d}/game_{}.pkl".format(num_of_games, int(time.time()))

    if not os.path.exists(os.path.dirname(filepath_in)):
        os.makedirs(os.path.dirname(filepath_in))

    generate_card_games(int(sys.argv[1]), filepath_in)

    for card in read_card_games(filepath_in):
        print(card)
