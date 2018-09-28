

import numpy as np

from collections import defaultdict

from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT


def random_choose(cards, trick, is_hearts_broken=False, is_pig_card_taken=False, is_double_taken=False):
    candicated_cards = []

    for suit, ranks in cards.items():
        bit_mask = 1
        while bit_mask <= 4096:
            if ranks & bit_mask:
                candicated_cards.append([suit, bit_mask])

            bit_mask <<= 1

    return candicated_cards, candicated_cards[np.random.choice(len(candicated_cards))]



def greedy_choose(cards, trick, is_hearts_broken=False, is_pig_card_taken=False, is_double_taken=False):
    def choose_max_card(cards, suits):
        candicated_cards = []

        bit_mask = 4096
        while bit_mask > 0:
            for suit in suits:
                rank = cards.get(suit, 0)
                if rank & bit_mask:
                    candicated_cards.append([suit, bit_mask])

            bit_mask >>= 1

        return candicated_cards


    def choose_min_card(cards, suits, is_pig_card_taken, own_pig_card):
        played_card = None

        bit_mask = 1
        while bit_mask <= 4096:
            for suit in suits:
                rank = cards.get(suit, 0)
                if rank & bit_mask:
                    if played_card is None:
                        played_card = [suit, bit_mask]

                        return played_card

            bit_mask <<= 1

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
            bit_mask = 4096
            while bit_mask > 0:
                if rank & bit_mask:
                    if bit_mask < max_rank:
                        if safe_play is None: safe_play = [leading_suit, bit_mask]
                    else:
                        if eaten_play is None: eaten_play = [leading_suit, bit_mask]


                bit_mask >>= 1

            if safe_play is None:
                safe_play = eaten_play
        else:
            if own_pig_card:
                safe_play = [SUIT_TO_INDEX["S"], NUM_TO_INDEX["Q"]]
            elif is_pig_card_taken == False and not own_pig_card and cards.get(SUIT_TO_INDEX["S"], 0) >= NUM_TO_INDEX["K"]:
                safe_play = [SUIT_TO_INDEX["S"], NUM_TO_INDEX["A"]]  if cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["A"] else [SUIT_TO_INDEX["S"], NUM_TO_INDEX["K"]]
            else:
                candicated_cards = choose_max_card(cards, suits=[SUIT_TO_INDEX["H"], SUIT_TO_INDEX["S"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["C"]])

                for suit, rank in sorted(candicated_cards, key=lambda x: -(x[1] << 2 if x[0] == SUIT_TO_INDEX["H"] else x[1])):
                    safe_play = [suit, rank]

                    break
    else:
        suits = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["S"], SUIT_TO_INDEX["H"]]
        if own_pig_card:
            suits = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["H"], SUIT_TO_INDEX["S"]]

        safe_play = choose_min_card(cards, suits, is_pig_card_taken, own_pig_card)

    return candicated_cards, safe_play


def expert_choose(cards, trick, is_hearts_broken=False, is_pig_card_taken=False, is_double_taken=False):
    def choose_max_card(cards, suits):
        candicated_cards = []

        bit_mask = 4096
        while bit_mask > 0:
            for suit in suits:
                rank = cards.get(suit, 0)
                if rank & bit_mask:
                    candicated_cards.append([suit, bit_mask])

            bit_mask >>= 1

        return candicated_cards


    def choose_min_card(cards, suits, is_pig_card_taken, own_pig_card):
        print(111111)
        candicated_cards, played_card = [], None

        num_of_suits = defaultdict(int)

        bit_mask = 1
        while bit_mask <= 4096:
            for suit in suits:
                if cards.get(suit, 0) & bit_mask:
                    num_of_suits[suit] += 1

            bit_mask <<= 1

        second_suit = None
        for suit_idx, (best_suit, num) in enumerate(sorted(num_of_suits.items(), key=lambda x: x[1])):
            if own_pig_card and best_suit == SUIT_TO_INDEX["S"]:
                continue

            if suit_idx == 1:
                second_suit = best_suit

            if second_suit is not None:
                break

        print("suit", best_suit, second_suit)

        if best_suit == SUIT_TO_INDEX["S"]:
            bit_mask = 1
            while bit_mask <= NUM_TO_INDEX["J"]:
               if cards[best_suit] & bit_mask:
                   played_card = [best_suit, bit_mask]

                   break

               bit_mask <<= 1

        if played_card is None:
            second_suit = second_suit if second_suit else best_suit

            bit_mask = 1
            while bit_mask <= 4096:
               if cards[second_suit] & bit_mask:
                   played_card = [second_suit, bit_mask]

                   break

               bit_mask <<= 1

        return played_card


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
            bit_mask = 4096
            while bit_mask > 0:
                if rank & bit_mask:
                    if bit_mask < max_rank:
                        if safe_play is None: safe_play = [leading_suit, bit_mask]
                    else:
                        if eaten_play is None: eaten_play = [leading_suit, bit_mask]


                bit_mask >>= 1

            if safe_play is None:
                safe_play = eaten_play
        else:
            if own_pig_card:
                safe_play = [SUIT_TO_INDEX["S"], NUM_TO_INDEX["Q"]]
            elif is_pig_card_taken == False and not own_pig_card and cards.get(SUIT_TO_INDEX["S"], 0) >= NUM_TO_INDEX["K"]:
                safe_play = [SUIT_TO_INDEX["S"], NUM_TO_INDEX["A"]]  if cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["A"] else [SUIT_TO_INDEX["S"], NUM_TO_INDEX["K"]]
            else:
                candicated_cards = choose_max_card(cards, suits=[SUIT_TO_INDEX["H"], SUIT_TO_INDEX["S"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["C"]])

                for suit, rank in sorted(candicated_cards, key=lambda x: -(x[1] << 2 if x[0] == SUIT_TO_INDEX["H"] else x[1])):
                    safe_play = [suit, rank]

                    break
    else:
        print(222222)
        suits = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["S"], SUIT_TO_INDEX["H"]]
        if own_pig_card:
            suits = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["H"], SUIT_TO_INDEX["S"]]

        safe_play = choose_min_card(cards, suits, is_pig_card_taken, own_pig_card)

    return candicated_cards, safe_play
