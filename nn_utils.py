#!/usr/bin/env python

import numpy as np

from card import Suit, Rank, Card


def card2v(card):
    return card.suit.value*13+card.rank.value

def v2card(v):
    suit, suit_num = None, v//13
    if suit_num == 0:
        suit = Suit.clubs
    elif suit_num == 1:
        suit = Suit.diamonds
    elif suit_num == 2:
        suit = Suit.spades
    else:
        suit = Suit.hearts

    rank, rank_num = None, v%15
    if rank_num == 2:
        rank = Rank.two
    elif rank_num == 3:
        rank = Rank.three
    elif rank_num == 4:
        rank = Rank.four
    elif rank_num == 5:
        rank = Rank.five
    elif rank_num == 6:
        rank = Rank.six
    elif rank_num == 7:
        rank = Rank.seven
    elif rank_num == 8:
        rank = Rank.eight
    elif rank_num == 9:
        rank = Rank.nine
    elif rank_num == 10:
        rank = Rank.ten
    elif rank_num == 11:
        rank = Rank.jack
    elif rank_num == 12:
        rank = Rank.queen
    elif rank_num == 13:
        rank = Rank.king
    elif rank_num == 14:
        rank = Rank.ace
    else:
        print("error_v:", v)
        raise

    return Card(suit, rank)


def played_prob_to_v(played_prob, n_slot=12):
    v = []

    for idx, (card, prob) in enumerate(played_prob):
        v.append([card2v(card), prob])

    for _ in range(n_slot-len(v)):
        v.append([0, 0])

    cards, probs = [], []
    for card, prob in v:
        cards.append(card)
        probs.append(prob)

    return cards, probs
