import copy

from random import choice
from functools import cmp_to_key
from collections import defaultdict

from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import bitmask_to_str


ALL_SUITS = [SUIT_TO_INDEX["C"], SUIT_TO_INDEX["D"], SUIT_TO_INDEX["S"], SUIT_TO_INDEX["H"]]


def choose_max_card(cards, suits, max_rank=NUM_TO_INDEX["A"], except_rank=0):
    bitmask = max_rank
    while bitmask >= NUM_TO_INDEX["2"]:
        if bitmask != except_rank:
            for suit in suits:
                if cards.get(suit, 0) & bitmask:
                    return [suit, bitmask]

        bitmask >>= 1


def choose_min_card(cards, suits, min_rank=NUM_TO_INDEX["2"], except_rank=0):
    bitmask = min_rank
    while bitmask <= NUM_TO_INDEX["A"]:
        if bitmask != except_rank:
            for suit in suits:
                if cards.get(suit, 0) & bitmask:
                    return [suit, bitmask]

        bitmask <<= 1


def numeric_compare(x, y):
    if x[1][0] == y[1][0]:
        return y[1][1] - x[1][1]
    else:
        return x[1][0] - y[1][0]


def choose_suit_card(cards, num_of_suits, suits, is_pig_card_taken, is_double_card_taken, own_pig_card, game_info, void_info):
    played_card = None

    is_all_long, score_size, remaining_size = True, defaultdict(int), defaultdict(tuple)
    for suit, num in sorted(num_of_suits.items(), key=lambda x: x[1]):
        size = game_info[suit][0]
        if num > 0:
            is_all_long &= (num>3)
            #print(suit, num)

            if num > 3:
                score_size[suit] = cards.get(suit, 0)-game_info[suit][1]

            if size > 0:
                remaining_size[suit] = (num, size)


    iterators = None
    if is_all_long:
        iterators = sorted(score_size.items(), key=lambda x: x[1])
    else:
        iterators = sorted(remaining_size.items(), key=cmp_to_key(numeric_compare))


    played_card, candicated_cards = None, []
    for suit, _ in iterators:
        if played_card is not None:
            break

        if suit is None:
            continue

        bitmask = NUM_TO_INDEX["2"]
        while bitmask <= game_info[suit][1]:
            rank = cards.get(suit, 0)
            if rank & bitmask:
                break

            bitmask <<= 1
        else:
            continue

        if suit == SUIT_TO_INDEX["S"]:
            if own_pig_card:
                if not void_info[SUIT_TO_INDEX["S"]] and cards.get(SUIT_TO_INDEX["S"], 0) >= NUM_TO_INDEX["K"] and num_of_suits[suit] > 3:
                    return play_spades_K_A(cards)

                continue

        if void_info[suit]:
            bitmask = NUM_TO_INDEX["2"]
            while bitmask <= NUM_TO_INDEX["A"]:
                if cards[suit] & bitmask:
                    played_card = [suit, bitmask]

                    break

                bitmask <<= 1
        else:
            bitmask = None

            if suit == SUIT_TO_INDEX["C"] and not is_double_card_taken:
                bitmask = NUM_TO_INDEX["9"]
            elif suit == SUIT_TO_INDEX["S"] and not is_pig_card_taken:
                bitmask = NUM_TO_INDEX["J"]
            else:
                bitmask = NUM_TO_INDEX["A"]

            while bitmask >= NUM_TO_INDEX["2"]:
                if cards[suit] & bitmask and bitmask < game_info[suit][1]:
                    played_card = [suit, bitmask]

                    break

                bitmask >>= 1

    #print("--->", played_card)
    return played_card


def get_num_for_suits(cards):
    global ALL_SUITS

    num_of_suits = defaultdict(int)
    valid_cards = []

    bitmask = NUM_TO_INDEX["2"]
    while bitmask <= NUM_TO_INDEX["A"]:
        for suit in ALL_SUITS:
            if cards.get(suit, 0) & bitmask:
                num_of_suits[suit] += 1

                valid_cards.append([suit, bitmask])

        bitmask <<= 1

    return num_of_suits, valid_cards


def possible_void(cards, num_of_suits, game_info):
    global ALL_SUITS

    void_info = dict([[suit, False] for suit in ALL_SUITS])
    for suit in ALL_SUITS:
        void_info[suit] = (game_info[suit][0] - num_of_suits[suit] < 4)

    return void_info


def play_spades_K_A(cards):
    return [SUIT_TO_INDEX["S"], NUM_TO_INDEX["A"]]  if cards[SUIT_TO_INDEX["S"]] & NUM_TO_INDEX["A"] else [SUIT_TO_INDEX["S"], NUM_TO_INDEX["K"]]


def expert_choose(position, cards, trick, real_own_pig_card, 
                  is_hearts_broken=False, is_pig_card_taken=False, is_double_card_taken=False, players_with_point=set(), game_info=None, void_info={}):

    num_of_suits, valid_cards = get_num_for_suits(cards)
    if len(valid_cards) == 1:
        return valid_cards, valid_cards[0]

    own_pig_card = (cards.get(SUIT_TO_INDEX["S"], 0) & NUM_TO_INDEX["Q"] > 0)

    possible_void_info = possible_void(cards, num_of_suits, game_info)

    void_info_for_suits = defaultdict(bool)
    for player_idx, info in void_info.items():
        if player_idx != position:
            for suit, is_void in info.items():
                void_info_for_suits[suit] |= (is_void or possible_void_info[suit])

    safe_play, candicated_cards = None, []
    if trick:
        len_trick, has_pig_card, has_double_card, has_point_card, leading_suit, max_rank = \
            len(trick), False, False, False, trick[0][0], trick[0][1]

        for suit, rank in trick:
            if suit == leading_suit and rank > max_rank:
                max_rank = rank

            has_pig_card |= (suit == SUIT_TO_INDEX["S"] and rank == NUM_TO_INDEX["Q"])
            has_point_card |= (has_pig_card or (suit == SUIT_TO_INDEX["H"]))
            has_double_card |= (suit == SUIT_TO_INDEX["C"] and rank == NUM_TO_INDEX["T"])

        eaten_play = None
        rank = cards.get(leading_suit, 0)

        trick_nr = 12-sum([info[0] for info in game_info])//4
        if rank > 0:
            if leading_suit == SUIT_TO_INDEX["S"]:
                if own_pig_card:
                    if max_rank > NUM_TO_INDEX["Q"]:
                        safe_play = [leading_suit, NUM_TO_INDEX["Q"]]
                    else:
                        safe_play = choose_max_card(cards, suits=[leading_suit], except_rank=NUM_TO_INDEX["Q"])
                else:
                    if len_trick == 3 and not has_point_card:
                        if cards.get(SUIT_TO_INDEX["S"], 0) >= NUM_TO_INDEX["K"]:
                            safe_play = play_spades_K_A(cards)

                        if safe_play is None:
                            safe_play = choose_max_card(cards, suits=[leading_suit], max_rank=NUM_TO_INDEX["J"])
                    else:
                        if void_info_for_suits[leading_suit]:
                            safe_play = choose_min_card(cards, suits=[leading_suit])
                        else:
                            safe_play = choose_max_card(cards, suits=[leading_suit], max_rank=NUM_TO_INDEX["J"])
            elif leading_suit == SUIT_TO_INDEX["C"]:
                if len_trick == 3 and not has_point_card and not has_double_card:
                    safe_play = choose_max_card(cards, suits=[leading_suit], max_rank=NUM_TO_INDEX["J"])

                if safe_play is None:
                    if void_info_for_suits[leading_suit] or has_pig_card:
                        safe_play = choose_min_card(cards, suits=[leading_suit])
                    else:
                        safe_play = choose_max_card(cards, suits=[leading_suit], max_rank=(NUM_TO_INDEX["A"] if is_double_card_taken or cards[leading_suit] < max_rank else NUM_TO_INDEX["9"]))
            elif leading_suit == SUIT_TO_INDEX["H"]:
                if trick_nr > 8 and len(players_with_point-set([position])) == 1 and cards[leading_suit] > max_rank:
                    safe_play = choose_max_card(cards, suits=[leading_suit])
                else:
                    safe_play = choose_min_card(cards, suits=[leading_suit])

            if safe_play is None:
                bitmask = NUM_TO_INDEX["A"]
                while bitmask >= NUM_TO_INDEX["2"]:
                    if rank & bitmask:
                        if bitmask < max_rank:
                            if safe_play is None:
                                safe_play = [leading_suit, bitmask]
                        else:
                            if eaten_play is None or has_pig_card or has_double_card:
                                eaten_play = [leading_suit, bitmask]

                    bitmask >>= 1

                if safe_play is None:
                    safe_play = eaten_play
        else:
            if trick_nr < 6 or len(players_with_point) > 1:
                if own_pig_card:
                    safe_play = [SUIT_TO_INDEX["S"], NUM_TO_INDEX["Q"]]
                elif is_pig_card_taken == False and cards.get(SUIT_TO_INDEX["S"], 0) >= NUM_TO_INDEX["K"]:
                    safe_play = play_spades_K_A(cards)

            if safe_play is None:
                remaining_size = defaultdict(tuple)
                for suit, num in sorted(num_of_suits.items(), key=lambda x: x[1]):
                    size = game_info[suit][0] - num

                    if size > 0:
                        remaining_size[suit] = (num, size)

                for suit in [suit for suit, _ in sorted(remaining_size.items(), key=cmp_to_key(numeric_compare))]:
                    if trick_nr > 6 and len(players_with_point-set([position])) == 1:
                        safe_play = choose_min_card(cards, suits=[suit])
                    else:
                        safe_play = choose_max_card(cards, suits=[suit])

                    if safe_play is not None:
                        break
    else:
        for suit, num in num_of_suits.items():
            game_info[suit][0] -= num
            game_info[suit][1] ^= cards.get(suit, 0)

        if len(players_with_point) == 1 and list(players_with_point)[0] == position:
            total_num, big_cards = 0, []
            for suit, num in num_of_suits.items():
                hearts_big_cards = []

                bitmask = NUM_TO_INDEX["2"]
                while bitmask <= NUM_TO_INDEX["A"]:
                    if cards.get(suit, 0) & bitmask and bitmask > game_info[suit][1]:
                        big_cards.append([suit, bitmask])

                        #if suit == SUIT_TO_INDEX["H"]:
                        hearts_big_cards.append([suit, bitmask])

                    bitmask <<= 1

                total_num += num

                if len(hearts_big_cards) > 0 and is_pig_card_taken and len(hearts_big_cards) >= game_info[suit][0]:
                    safe_play = choice(hearts_big_cards)

                    break

            if len(big_cards) > total_num*0.66667:
                safe_play = choice(big_cards)


        if safe_play is None:
            suits = ALL_SUITS
            safe_play = choose_suit_card(cards, num_of_suits, suits, is_pig_card_taken, is_double_card_taken, real_own_pig_card, game_info, void_info_for_suits)


    for suit, num in sorted(num_of_suits.items(), key=lambda x: x[1]):
        if safe_play is not None:
            break

        if num == 0:
            continue

        bitmask = NUM_TO_INDEX["2"]
        while bitmask <= NUM_TO_INDEX["A"]:
            if cards.get(suit, 0) & bitmask:
                safe_play = [suit, bitmask]

                break

            bitmask <<= 1

    return candicated_cards, safe_play
