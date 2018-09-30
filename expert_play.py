from collections import defaultdict

from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import bitmask_to_str


ALL_SUITS = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["S"], SUIT_TO_INDEX["H"]]


def additional_score(suit, rank):
    if suit == SUIT_TO_INDEX["H"]:
        return rank << 3
    elif suit == SUIT_TO_INDEX["C"] and rank & NUM_TO_INDEX["T"]:
        return rank << 3
    else:
        return rank


def choose_max_card(cards, suits, max_rank=NUM_TO_INDEX["A"]):
    bit_mask = max_rank
    while bit_mask >= NUM_TO_INDEX["2"]:
        for suit in suits:
            if cards.get(suit, 0) & bit_mask:
                return [suit, bit_mask]

        bit_mask >>= 1


def choose_min_card(cards, num_of_suits, suits, is_pig_card_taken, own_pig_card, game_info, void_info):
    played_card = None

    best_suit, second_suit = None, None
    for suit_idx, (suit, num) in enumerate(sorted(num_of_suits.items(), key=lambda x: x[1])):
        if num > 0 and best_suit is None:
            best_suit = suit

            continue

        if num > 0 and second_suit is None:
            second_suit = suit

            break

    if best_suit == SUIT_TO_INDEX["S"]:
        if own_pig_card:
            if not void_info[SUIT_TO_INDEX["S"]] and cards.get(SUIT_TO_INDEX["S"], 0) >= NUM_TO_INDEX["K"]:
                return play_spades_K_A(cards)

        if not is_pig_card_taken and second_suit is not None:
            best_suit, second_suit = second_suit, None

    #print("best_suit, second_suit = ({}, {})".format(best_suit, second_suit))

    played_card = None
    for suit in [best_suit, second_suit]:
        if void_info[suit]:
            bit_mask = NUM_TO_INDEX["2"]
            while bit_mask <= NUM_TO_INDEX["A"]:
                if cards[suit] & bit_mask and bit_mask < game_info[suit][1]:
                    played_card = [suit, bit_mask]

                    break

                bit_mask <<= 1
        else:
            bit_mask = NUM_TO_INDEX["A"]
            while bit_mask >= NUM_TO_INDEX["2"]:
                if cards[best_suit] & bit_mask:
                    played_card = [best_suit, bit_mask]

                    break

                bit_mask >>= 1

    return played_card


def get_num_for_suits(cards):
    global ALL_SUITS

    num_of_suits = defaultdict(int)

    bit_mask = NUM_TO_INDEX["2"]
    while bit_mask <= NUM_TO_INDEX["A"]:
        for suit in ALL_SUITS:
            if cards.get(suit, 0) & bit_mask:
                num_of_suits[suit] += 1

        bit_mask <<= 1

    return num_of_suits


def possible_void(cards, num_of_suits, game_info):
    global ALL_SUITS

    void_info = dict([[suit, False] for suit in ALL_SUITS])
    for suit in ALL_SUITS:
        void_info[suit] = (game_info[suit][0] - num_of_suits[suit] < 4)

    return void_info


def play_spades_K_A(cards):
    return [SUIT_TO_INDEX["S"], NUM_TO_INDEX["A"]]  if cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["A"] else [SUIT_TO_INDEX["S"], NUM_TO_INDEX["K"]]


def expert_choose(position, cards, trick, is_hearts_broken=False, is_pig_card_taken=False, is_double_taken=False, players_with_point=set(), game_info=None, void_info={}):
    own_pig_card = cards.get(SUIT_TO_INDEX["S"], 0) & NUM_TO_INDEX["Q"]

    num_of_suits = get_num_for_suits(cards)

    possible_void_info = possible_void(cards, num_of_suits, game_info)

    void_info_for_suits = defaultdict(bool)
    for player_idx, info in void_info.items():
        if player_idx != position:
            for suit, is_void in info.items():
                void_info_for_suits[suit] |= (is_void or possible_void_info[suit])

    #print("num_of_suits", num_of_suits)
    #print("void_info", position, void_info)
    #print("possible_void_info", position, possible_void_info)

    safe_play, candicated_cards = None, []
    if trick:
        leading_suit, max_rank = trick[0][0], trick[0][1]

        for suit, rank in trick[1:]:
            if suit == leading_suit and rank > max_rank:
                max_rank = rank

        eaten_play = None
        rank = cards.get(leading_suit, 0)

        if rank > 0:
            """
            if leading_suit == SUIT_TO_INDEX["S"]:
                if own_pig_card:
                    if max_rank > NUM_TO_INDEX["Q"]:
                        return [leading_suit, NUM_TO_INDEX["Q"]]
                    else:
                        return choose_max_card(cards, suits=suits[:1], max_rank=NUM_TO_INDEX["J"])[0]
                elif not is_pig_card_taken:
                    if void_info_for_suits[leading_suit]:

                    else:
                        return choose_max_card(cards, suits=suits[:1], max_rank=NUM_TO_INDEX["J"])[0]
            """

            bit_mask = NUM_TO_INDEX["A"]
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
                safe_play = play_spades_K_A(cards)
            else:
                for suit in [suit for suit, num in sorted(num_of_suits.items(), key=lambda x: x[1])]:
                    safe_play = choose_max_card(cards, suits=[suit])

                    if safe_play is not None:
                        break

                #for suit, rank in sorted(candicated_cards, key=lambda x: -additional_score(x[0], x[1])):
                #    safe_play = [suit, rank]

                #    break
    else:
        suits = ALL_SUITS
        safe_play = choose_min_card(cards, num_of_suits, suits, is_pig_card_taken, own_pig_card, game_info, void_info_for_suits)

    return candicated_cards, safe_play
