import numpy as np

from collections import defaultdict

from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import bitmask_to_str


def additional_score(suit, rank):
    if suit == SUIT_TO_INDEX["H"]:
        return rank << 3
    elif suit == SUIT_TO_INDEX["C"] and rank & NUM_TO_INDEX["T"]:
        return rank << 3
    else:
        return rank


def choose_max_card(cards, suits):
    candicated_cards = []

    bit_mask = NUM_TO_INDEX["A"]
    while bit_mask > 0:
        for suit in suits:
            rank = cards.get(suit, 0)
            if rank & bit_mask:
                candicated_cards.append([suit, bit_mask])

        bit_mask >>= 1

    return candicated_cards


def choose_min_card(cards, suits, is_pig_card_taken, own_pig_card):
    played_card = None

    bit_mask = NUM_TO_INDEX["2"]
    while bit_mask <= NUM_TO_INDEX["A"]:
        for suit in suits:
            if cards.get(suit, 0) & bit_mask:
                played_card = [suit, bit_mask]

                return played_card

        bit_mask <<= 1


def random_choose(position, cards, trick, own_pig_card, is_hearts_broken=False, is_pig_card_taken=False, is_double_taken=False, players_with_point=set(), game_info=None, void_info={}):
    candicated_cards = []

    for suit, ranks in cards.items():
        bit_mask = 1
        while bit_mask <= NUM_TO_INDEX["A"]:
            if ranks & bit_mask:
                candicated_cards.append([suit, bit_mask])

            bit_mask <<= 1

    return candicated_cards, candicated_cards[np.random.choice(len(candicated_cards))]


def greedy_choose(position, cards, trick, own_pig_card, is_hearts_broken=False, is_pig_card_taken=False, is_double_taken=False, players_with_point=set(), game_info=None, void_info={}):
    own_pig_card = cards.get(SUIT_TO_INDEX["S"], 0) & NUM_TO_INDEX["Q"]

    safe_play, candicated_cards = None, []
    if trick:
        leading_suit, max_rank = trick[0][0], trick[0][1]

        for suit, rank in trick[1:]:
            if suit == leading_suit and rank > max_rank:
                max_rank = rank

        eaten_play = None
        rank = cards.get(leading_suit, 0)

        if rank > 0:
            if leading_suit == SUIT_TO_INDEX["S"] and own_pig_card:
                if max_rank > NUM_TO_INDEX["Q"]:
                    safe_play = [leading_suit, NUM_TO_INDEX["Q"]]

            if safe_play is None:
                safe_plays, eaten_plays = [], []

                bit_mask = NUM_TO_INDEX["A"]
                while bit_mask >= NUM_TO_INDEX["2"]:
                    if rank & bit_mask:
                        if bit_mask < max_rank:
                            safe_plays.append([leading_suit, bit_mask])
                        else:
                            eaten_plays.append([leading_suit, bit_mask])

                    bit_mask >>= 1

                if safe_plays:
                    safe_play = sorted(safe_plays, key=lambda x: additional_score(x[0], x[1]))[-1]
                else:
                    safe_play = eaten_plays[0]
        else:
            if own_pig_card:
                safe_play = [SUIT_TO_INDEX["S"], NUM_TO_INDEX["Q"]]
            elif is_pig_card_taken == False and not own_pig_card and cards.get(SUIT_TO_INDEX["S"], 0) >= NUM_TO_INDEX["K"]:
                safe_play = [SUIT_TO_INDEX["S"], NUM_TO_INDEX["A"]]  if cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["A"] else [SUIT_TO_INDEX["S"], NUM_TO_INDEX["K"]]
            else:
                candicated_cards = choose_max_card(cards, suits=[SUIT_TO_INDEX["H"], SUIT_TO_INDEX["S"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["C"]])

                for suit, rank in sorted(candicated_cards, key=lambda x: -additional_score(x[0], x[1])):
                    safe_play = [suit, rank]

                    break
    else:
        suits = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["S"], SUIT_TO_INDEX["H"]]
        if own_pig_card:
            suits = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["H"], SUIT_TO_INDEX["S"]]

        safe_play = choose_min_card(cards, suits, is_pig_card_taken, own_pig_card)

    return candicated_cards, safe_play
