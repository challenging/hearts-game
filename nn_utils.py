#!/usr/bin/env python

import numpy as np

from card import Suit, Rank


def card2v(card):
    return card.suit.value*13+card.rank.value


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
