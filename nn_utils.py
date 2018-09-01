#!/usr/bin/env python

import numpy as np

from card import Suit, Rank


def card2v(card):
    v = [0, 0, 0, 0]
    v[card.suit.value] = card.rank.value

    return v


def played_prob_to_v(played_prob, n_slot=12):
    v = []

    for idx, (card, prob) in enumerate(played_prob):
        v.append([card2v(card), prob])

    for _ in range(n_slot-len(v)):
        v.append([[0, 0, 0, 0], 0])

    return v
