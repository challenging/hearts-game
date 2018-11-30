#!/usr/bin/env python

import numpy as np

from card import Suit, Rank, Card, Deck, SPADES_Q, CLUBS_T
from card import NUM_TO_INDEX, SUIT_TO_INDEX, INDEX_TO_NUM, INDEX_TO_SUIT
from card import bitmask_to_card, batch_bitmask_to_card


def card2v(card):
    return card.suit.value*13+(card.rank.value-2)

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

    rank, rank_num = None, v%13
    if rank_num == 0:
        rank = Rank.two
    elif rank_num == 1:
        rank = Rank.three
    elif rank_num == 2:
        rank = Rank.four
    elif rank_num == 3:
        rank = Rank.five
    elif rank_num == 4:
        rank = Rank.six
    elif rank_num == 5:
        rank = Rank.seven
    elif rank_num == 6:
        rank = Rank.eight
    elif rank_num == 7:
        rank = Rank.nine
    elif rank_num == 8:
        rank = Rank.ten
    elif rank_num == 9:
        rank = Rank.jack
    elif rank_num == 10:
        rank = Rank.queen
    elif rank_num == 11:
        rank = Rank.king
    elif rank_num == 12:
        rank = Rank.ace
    else:
        print("error_v:", v)
        raise

    return Card(suit, rank)


def get_index(card):
    return card.suit.value, card.rank.value-2


def transform_trick_cards(trick_cards):
    results = np.zeros([13, 4, 4, 13], dtype=np.int32)

    for round_idx, tricks in enumerate(trick_cards):
        for player_idx, card in enumerate(tricks):
            if card:
                suit_idx, rank_idx = get_index(card)
                results[round_idx:, player_idx, suit_idx, rank_idx] = 1

    return results


def transform_score_cards(score_cards):
    results = np.zeros([1, 4, 4, 13], dtype=np.int32)

    for player_idx, cards in enumerate(score_cards):
        for card in cards:
            suit_idx, rank_idx = get_index(card)
            results[0, player_idx, suit_idx, rank_idx] = 1

    return results


def transform_possible_cards(possible_cards):
    results = np.zeros([1, 4, 4, 13], dtype=np.int32)

    for player_idx, cards in enumerate(possible_cards):
        for card in cards:
            suit_idx, rank_idx = get_index(card)

            results[0, player_idx, suit_idx, rank_idx] = 1

    return results


def transform_this_trick_cards(current_player_idx, trick_cards):
    results = np.zeros([3, 4, 4, 13], dtype=np.int32)

    step = 0
    for idx, card in zip(range(3, 0, -1), trick_cards[::-1]):
        player_idx = (current_player_idx+idx)%4
        suit_idx, rank_idx = get_index(card)

        results[step, player_idx, suit_idx, rank_idx] = 1

        step += 1

    return results


def transform_valid_cards(current_player_idx, valid_cards):
    results = np.zeros([1, 4, 4, 13], dtype=np.int32)

    for card in valid_cards:
        suit_idx, rank_idx = get_index(card)

        results[0, current_player_idx, suit_idx, rank_idx] = 1

    return results


def transform_leading_cards(current_player_idx, is_leading):
    results = np.zeros([1, 4, 4, 13], dtype=np.int32)

    if is_leading:
        results[0,current_player_idx] = 1

    return results


def transform_expose_cards(expose_info):
    results = np.ones([1, 4, 4, 13], dtype=np.int32)

    if np.max(expose_info) == 2:
        results[0, np.argmax(expose_info)] = 2

    return results


def transform_results(score_cards):
    results = np.zeros([15, 4], dtype=np.int32)

    for player_idx, cards in enumerate(score_cards):
        for card in cards:
            if card == SPADES_Q:
                results[13, player_idx] = 1
            elif card == CLUBS_T:
                results[14, player_idx] = 1
            else:
                _, rank_idx = get_index(card)

                results[rank_idx, player_idx] = 1

    return results


def get_b_index(card):
    suit_idx, rank = card

    rank_idx = INDEX_TO_NUM[rank]
    if rank_idx.isdigit():
        rank_idx = int(rank_idx)
    elif rank_idx == "T":
        rank_idx = 10
    elif rank_idx == "J":
        rank_idx = 11
    elif rank_idx == "Q":
        rank_idx = 12
    elif rank_idx == "K":
        rank_idx = 13
    elif rank_idx == "A":
        rank_idx = 14
    else:
        raise

    return suit_idx, rank_idx-2


def transform_b_trick_cards(current_player_idx, trick_cards):
    results = np.zeros([13, 4, 4, 13], dtype=np.int32)
    for round_idx, cards in enumerate(trick_cards):
        for idx, card in zip(range(4, 0, -1), cards[::-1]):
            if card:
                player_idx = (current_player_idx+idx)%4
                suit_idx, rank_idx = get_b_index(card)

                results[round_idx:, player_idx, suit_idx, rank_idx] = 1

    return results


def transform_b_score_cards(score_cards):
    results = np.zeros([1, 4, 4, 13], dtype=np.int32)
    for player_idx, cards in enumerate(score_cards):
        for suit, ranks in enumerate(cards):
            bitmask = NUM_TO_INDEX["2"]
            while bitmask <= NUM_TO_INDEX["A"]:
                if ranks & bitmask:
                    suit_idx, rank_idx = get_b_index((suit, bitmask))
                    results[0, player_idx, suit_idx, rank_idx] = 1

                bitmask <<= 1

    return results


def transform_b_possible_cards(possible_cards):
    results = np.zeros([1, 4, 4, 13], dtype=np.int32)
    for player_idx, cards in enumerate(possible_cards):
        for suit, ranks in cards.items():
            bitmask = NUM_TO_INDEX["2"]
            while bitmask <= NUM_TO_INDEX["A"]:
                if ranks & bitmask:
                    suit_idx, rank_idx = get_b_index((suit, bitmask))
                    results[0, player_idx, suit_idx, rank_idx] = 1

                bitmask <<= 1

    return results


def transform_b_this_trick_cards(current_player_idx, this_trick_cards):
    results = np.zeros([3, 4, 4, 13], dtype=np.int32)
    step = 0
    for idx, card in zip(range(3, 0, -1), this_trick_cards[::-1]):
        player_idx = (current_player_idx+idx)%4
        suit_idx, rank_idx = get_b_index(card)

        results[step, player_idx, suit_idx, rank_idx] = 1

        step += 1

    return results


def transform_b_valid_cards(current_player_idx, valid_cards):
    results = np.zeros([1, 4, 4, 13], dtype=np.int32)
    for suit, ranks in valid_cards.items():
        bitmask = NUM_TO_INDEX["2"]
        while bitmask <= NUM_TO_INDEX["A"]:
            if ranks & bitmask:
                suit_idx, rank_idx = get_b_index((suit, bitmask))
                results[0, current_player_idx, suit_idx, rank_idx] = 1

            bitmask <<= 1

    return results


def transform_game_info_to_nn(state, trick_nr, is_debug=False):
    current_trick_nr = trick_nr+len(state.tricks)-1

    current_player_idx = state.start_pos

    trick_cards = transform_b_trick_cards(current_player_idx, state.trick_cards)
    score_cards = transform_b_score_cards(state.score_cards)
    possible_cards = transform_b_possible_cards(state.get_possible_cards())
    this_trick_cards = transform_b_this_trick_cards(state.start_pos, state.tricks[-1][1])
    valid_cards = transform_b_valid_cards(current_player_idx, state.get_valid_cards(state.hand_cards[current_player_idx], current_trick_nr, is_playout=True))
    leading_cards = transform_leading_cards(current_player_idx, len(state.tricks[-1][1]) == 0)
    expose_cards = transform_expose_cards(state.expose_info)

    if is_debug:
        a_memory = []
        a_memory.append(current_player_idx)
        a_memory.append(state.trick_cards)
        a_memory.append(state.score_cards)
        a_memory.append(state.get_possible_cards())
        a_memory.append(state.tricks[-1][1])
        a_memory.append(state.get_valid_cards(state.hand_cards[current_player_idx], current_trick_nr, is_playout=True))
        a_memory.append(len(state.tricks[-1][1]) == 0)
        a_memory.append(state.expose_info)

        print_a_b_memory(a_memory)

    return trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards


def print_a_b_memory(played_data):
    current_player_idx = played_data[0]
    trick_cards = played_data[1]
    score_cards = played_data[2]
    possible_cards = played_data[3]
    this_trick_cards = played_data[4]
    valid_cards = played_data[5]
    leading_cards = played_data[6]
    expose_cards = played_data[7]

    results_trick_cards = transform_b_trick_cards(current_player_idx, trick_cards)
    for round_idx, tricks in enumerate(trick_cards):
        print("      round_idx:", round_idx+1)
        for player_idx, card in enumerate(tricks):
            if card:
                print("    \ttrick_cards:", player_idx, card, np.where(results_trick_cards[round_idx:, player_idx] == 1))

    results_score_cards = transform_b_score_cards(score_cards)
    for player_idx, cards in enumerate(score_cards):
        print("    score_cards:", player_idx, cards, np.where(results_score_cards[0, player_idx] == 1))

    results_possible_cards = transform_b_possible_cards(possible_cards)
    for player_idx, cards in enumerate(possible_cards):
        print(" possible_cards:", player_idx, cards)
        print(" possible_cards:", player_idx, results_possible_cards[0, player_idx, ::])

    results_this_trick = transform_b_this_trick_cards(current_player_idx, this_trick_cards)
    for idx, card in zip(range(3, 0, -1), this_trick_cards[::-1]):
        player_idx = (current_player_idx+idx)%4
        print("    this_trick:", player_idx, card, np.where(results_this_trick[-idx, player_idx] == 1))

    results_valid_cards = transform_b_valid_cards(current_player_idx, valid_cards)
    print("   valid_cards:", current_player_idx, valid_cards, np.where(results_valid_cards[0, 0] == 1))

    results_leading_cards = transform_leading_cards(current_player_idx, leading_cards)
    print("    is_leading:", leading_cards, np.unique(results_leading_cards), results_leading_cards.shape)

    results_expose_cards = transform_expose_cards(expose_cards)
    print("    is_expose:", expose_cards, np.unique(results_expose_cards), results_expose_cards.shape)


def print_a_memory(played_data):
    current_player_idx = played_data[0]
    trick_cards = played_data[1]
    score_cards = played_data[2]
    possible_cards = played_data[3]
    this_trick_cards = played_data[4]
    valid_cards = played_data[5]
    leading_cards = played_data[6]
    expose_cards = played_data[7]
    probs = played_data[8]
    results = played_data[9]

    results_trick_cards = transform_trick_cards(trick_cards)
    for round_idx, tricks in enumerate(trick_cards):
        print("      round_idx:", round_idx+1)
        for player_idx, card in enumerate(tricks):
            if card:
                print("    \ttrick_cards:", player_idx, card, np.where(results_trick_cards[round_idx:, player_idx] == 1))

    results_score_cards = transform_score_cards(score_cards)
    for player_idx, cards in enumerate(score_cards):
        print("    score_cards:", player_idx, cards, np.where(results_score_cards[0, player_idx] == 1))

    results_possible_cards = transform_possible_cards(possible_cards)
    for player_idx, cards in enumerate(possible_cards):
        print(" possible_cards:", player_idx, cards)
        print(" possible_cards:", player_idx, results_possible_cards[0, player_idx, ::])

    results_this_trick = transform_this_trick_cards(current_player_idx, this_trick_cards)
    for idx, card in zip(range(3, 0, -1), this_trick_cards[::-1]):
        player_idx = (current_player_idx+idx)%4
        print("    this_trick:", player_idx, card, np.where(results_this_trick[-idx, player_idx] == 1))

    results_valid_cards = transform_valid_cards(current_player_idx, valid_cards)
    print("   valid_cards:", current_player_idx, valid_cards, np.where(results_valid_cards[0, 0] == 1))

    results_leading_cards = transform_leading_cards(current_player_idx, leading_cards)
    print("    is_leading:", leading_cards, np.unique(results_leading_cards), results_leading_cards.shape)

    results_expose_cards = transform_expose_cards(expose_cards)
    print("    is_expose:", expose_cards, np.unique(results_expose_cards), results_expose_cards.shape)

    if probs:
        print("        probs:", probs)

    if results:
        results_results = transform_results(results)
        for player_idx, sub_results in enumerate(results):
            print("  results:", player_idx, sub_results, np.where(results_results[player_idx*15:(player_idx+1)*15] == 1))

    print()


if __name__ == "__main__":
    from rules import transform_cards

    card_string = [["JH", "6H,8H", "AH", "TC,KH,4H,9H,QS,QH,5H,TH,3H,7H,2H"]]
    results = transform_cards(card_string)[0]
    print(results)

    print(transform_results(results))
