import copy

from functools import cmp_to_key
from collections import defaultdict

from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import bitmask_to_str


ALL_SUITS = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["S"], SUIT_TO_INDEX["H"]]


def choose_max_card(cards, suits, max_rank=NUM_TO_INDEX["A"], except_rank=0):
    bit_mask = max_rank
    while bit_mask >= NUM_TO_INDEX["2"]:
        if bit_mask != except_rank:
            for suit in suits:
                if cards.get(suit, 0) & bit_mask:
                    return [suit, bit_mask]

        bit_mask >>= 1


def choose_min_card(cards, suits, min_rank=NUM_TO_INDEX["2"], except_rank=0):
    bit_mask = min_rank
    while bit_mask <= NUM_TO_INDEX["A"]:
        if bit_mask != except_rank:
            for suit in suits:
                if cards.get(suit, 0) & bit_mask:
                    return [suit, bit_mask]

        bit_mask <<= 1


def numeric_compare(x, y):
    if x[1][0] == y[1][0]:
        return y[1][1] - x[1][1]
    else:
        return x[1][0] - y[1][0]


def choose_suit_card(cards, num_of_suits, suits, is_pig_card_taken, is_double_card_taken, own_pig_card, game_info, void_info):
    played_card = None

    remaining_size = defaultdict(tuple)
    best_suit, second_suit, third_suit = None, None, None
    for suit, num in sorted(num_of_suits.items(), key=lambda x: x[1]):
        size = game_info[suit][0] - num
        if num > 0 and size > 0:
            game_info[suit][0] -= num
            game_info[suit][1] ^= cards[suit]

            remaining_size[suit] = (num, size)
            if best_suit is None:
                best_suit = suit
            elif second_suit is None:
                second_suit = suit
            elif third_suit is None:
                third_suit = suit

                break


    #print(best_suit, second_suit, third_suit, cards.get(SUIT_TO_INDEX["S"], 0), num_of_suits, remaining_size)

    played_card, candicated_cards = None, []
    for suit, _ in sorted(remaining_size.items(), key=cmp_to_key(numeric_compare)):
        if played_card is not None:
            break

        if suit is None:
            continue

        bit_mask = NUM_TO_INDEX["2"]
        while bit_mask <= game_info[suit][1]:
            rank = cards.get(suit, 0)
            if rank & bit_mask:
                break

            bit_mask <<= 1
        else:
            continue

        if suit == SUIT_TO_INDEX["S"]:
            if own_pig_card:
                if not void_info[SUIT_TO_INDEX["S"]] and cards.get(SUIT_TO_INDEX["S"], 0) >= NUM_TO_INDEX["K"] and num_of_suits[suit] > 3:
                    return play_spades_K_A(cards)

                continue

        if void_info[suit] or suit == SUIT_TO_INDEX["H"]:
            bit_mask = NUM_TO_INDEX["2"]
            while bit_mask <= NUM_TO_INDEX["A"]:
                if cards[suit] & bit_mask:
                    played_card = [suit, bit_mask]

                    break

                bit_mask <<= 1
        else:
            bit_mask = None

            if suit == SUIT_TO_INDEX["C"] and not is_double_card_taken:
                bit_mask = NUM_TO_INDEX["9"]
            elif suit == SUIT_TO_INDEX["S"] and not is_pig_card_taken:
                bit_mask = NUM_TO_INDEX["J"]
            else:
                bit_mask = NUM_TO_INDEX["A"]

            while bit_mask >= NUM_TO_INDEX["2"]:
                if cards[suit] & bit_mask:
                    played_card = [suit, bit_mask]

                    break

                bit_mask >>= 1

    #print("--->", played_card)
    for suit, num in sorted(num_of_suits.items(), key=lambda x: x[1]):
        if played_card is not None:
            break

        if num == 0:
            continue

        bit_mask = NUM_TO_INDEX["2"]
        while bit_mask <= NUM_TO_INDEX["A"]:
            if cards.get(suit, 0) & bit_mask:
                played_card = [suit, bit_mask]

                break

            bit_mask <<= 1

    #print("--->", played_card, cards)
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


def expert_choose(position, cards, trick, real_own_pig_card, is_hearts_broken=False, is_pig_card_taken=False, is_double_card_taken=False, players_with_point=set(), game_info=None, void_info={}):
    own_pig_card = (cards.get(SUIT_TO_INDEX["S"], 0) & NUM_TO_INDEX["Q"] > 0)
    num_of_suits = get_num_for_suits(cards)

    possible_void_info = possible_void(cards, num_of_suits, game_info)

    void_info_for_suits = defaultdict(bool)
    for player_idx, info in void_info.items():
        if player_idx != position:
            for suit, is_void in info.items():
                void_info_for_suits[suit] |= (is_void or possible_void_info[suit])

    #print("num_of_suits", position, num_of_suits, own_pig_card)
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
            if leading_suit == SUIT_TO_INDEX["S"]:
                if own_pig_card:
                    if max_rank > NUM_TO_INDEX["Q"]:
                        safe_play = [leading_suit, NUM_TO_INDEX["Q"]]
                    else:
                        safe_play = choose_max_card(cards, suits=[leading_suit], except_rank=NUM_TO_INDEX["Q"])
                else:
                    if void_info_for_suits[leading_suit]:
                        safe_play = choose_min_card(cards, suits=[leading_suit])
                    else:
                        safe_play = choose_max_card(cards, suits=[leading_suit], max_rank=NUM_TO_INDEX["J"])
            elif leading_suit == SUIT_TO_INDEX["C"]:
                if void_info_for_suits[leading_suit]:
                    safe_play = choose_min_card(cards, suits=[leading_suit])
                else:
                    safe_play = choose_max_card(cards, suits=[leading_suit], max_rank=(NUM_TO_INDEX["A"] if is_double_card_taken or cards[leading_suit] < max_rank else NUM_TO_INDEX["T"]))
            elif leading_suit == SUIT_TO_INDEX["H"] and len(players_with_point) < 2:
                safe_play = choose_min_card(cards, suits=[leading_suit])

            if safe_play is None:
                bit_mask = NUM_TO_INDEX["A"]
                while bit_mask >= NUM_TO_INDEX["2"]:
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
            elif is_pig_card_taken == False and cards.get(SUIT_TO_INDEX["S"], 0) >= NUM_TO_INDEX["K"]:
                safe_play = play_spades_K_A(cards)
            else:
                remaining_size = defaultdict(tuple)
                for suit, num in sorted(num_of_suits.items(), key=lambda x: x[1]):
                    remaining_size[suit] = (num, game_info[suit][0] - num)

                for suit in [suit for suit, _ in sorted(remaining_size.items(), key=cmp_to_key(numeric_compare))]:
                    safe_play = choose_max_card(cards, suits=[suit])

                    if safe_play is not None:
                        break
    else:
        suits = ALL_SUITS
        safe_play = choose_suit_card(cards, num_of_suits, suits, is_pig_card_taken, is_double_card_taken, real_own_pig_card, copy.deepcopy(game_info), void_info_for_suits)

    return candicated_cards, safe_play
