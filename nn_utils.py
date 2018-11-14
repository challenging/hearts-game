#!/usr/bin/env python

from card import Suit, Rank, Card, Deck, NUM_TO_INDEX
from card import bitmask_to_card, batch_bitmask_to_card


NONE_CARD_INDEX = 0

def card2v(card):
    return card.suit.value*13+(card.rank.value-2)+1

def v2card(v):
    v -= 1

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


def full_cards(cards):
    results = []

    if isinstance(cards, list):
        for card in Deck().cards:
            if card in cards:
                results.append(card2v(card))
            else:
                results.append(NONE_CARD_INDEX)
    elif isinstance(cards, dict):
        for card in Deck().cards:
            results.append(cards[card])

    return results


def limit_cards(cards, num_slot):
    results = []

    for card in cards:
        results.append(card2v(card))

    for _ in range(num_slot-len(results)):
        results.append(NONE_CARD_INDEX)

    return results


def transform_game_info_to_nn(state, trick_nr):
    a_memory = []

    remaining_cards = []
    for suit, info in enumerate(state.current_info):
        remaining_cards.extend(list(batch_bitmask_to_card(suit, info[1])))
    a_memory.append(remaining_cards)
    #remaining_cards = full_cards(remaining_cards)

    trick_cards = [bitmask_to_card(suit, rank) for suit, rank in state.tricks[-1][1]]
    a_memory.append(trick_cards)

    must_cards = [[], [], [], []]
    for player_idx in range(4):
        must_cards[player_idx] = state.must_have.get(player_idx, [])
    a_memory.append(must_cards)

    score_cards = [[], [], [], []]
    for player_idx, cards in enumerate(state.score_cards):
        for suit, ranks in enumerate(cards):
            score_cards[player_idx].extend(list(batch_bitmask_to_card(suit, ranks)))

        #score_cards[player_idx] = full_cards(score_cards[player_idx])
    a_memory.append(score_cards)

    hand_cards = []
    for suit, ranks in state.hand_cards[state.start_pos].items():
        bitmask = NUM_TO_INDEX["2"]
        while bitmask <= NUM_TO_INDEX["A"]:
            if ranks & bitmask:
                hand_cards.append(bitmask_to_card(suit, bitmask))

            bitmask <<= 1
    a_memory.append(hand_cards)

    valid_cards = []
    for suit, ranks in state.get_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1, is_playout=True).items():
        bitmask = NUM_TO_INDEX["2"]
        while bitmask <= NUM_TO_INDEX["A"]:
            if ranks & bitmask:
                valid_cards.append(bitmask_to_card(suit, bitmask))

            bitmask <<= 1
    a_memory.append(valid_cards)

    #print_a_memory(a_memory)

    return full_cards(remaining_cards), \
           limit_cards(trick_cards, 3), \
           limit_cards(must_cards[0], 4), limit_cards(must_cards[1], 4), limit_cards(must_cards[2], 4), limit_cards(must_cards[3], 4), \
           full_cards(score_cards[0]), full_cards(score_cards[1]), full_cards(score_cards[2]), full_cards(score_cards[3]), \
           limit_cards(hand_cards, 13), limit_cards(valid_cards, 13)


def print_a_memory(played_data):
    remaining_cards = played_data[0]
    trick_cards = played_data[1]
    must_cards = played_data[2]
    score_cards = played_data[3]
    hand_cards = played_data[4]
    valid_cards = played_data[5]
    played_cards = played_data[6] if len(played_data) > 6 else None
    probs = played_data[7] if len(played_data) > 7 else None
    score = played_data[8] if len(played_data) > 8 else None

    print("remaining_cards:", [(card, card2v(card)) for card in remaining_cards])
    print("remaining_cards:", full_cards(remaining_cards))

    print("    trick_cards:", [(card, card2v(card)) for card in trick_cards])
    print("    trick_cards:", limit_cards(trick_cards, 4))

    print("     must_cards:", [[(card, card2v(card)) for card in cards] for cards in must_cards])
    print("     must_cards:", [limit_cards(cards, 4) for cards in must_cards])

    print("    score_cards:", [[(card, card2v(card)) for card in cards] for cards in score_cards])
    print("    score_cards:", [limit_cards(cards, 52) for cards in score_cards])

    print("     hand_cards:", hand_cards)
    print("     hand_cards:", limit_cards(hand_cards, 13))

    print("    valid_cards:", valid_cards)
    print("    valid_cards:", limit_cards(valid_cards, 13))

    if played_cards:
        print("   played_cards:", [(card, card2v(card)) for card in played_cards])
        print("          probs:", probs)
        print("          probs:", limit_cards(dict(zip(played_cards, probs))), 13)
        print("          score:", score)
        print()


if __name__ == "__main__":
    for card in Deck().cards:
        print(card, card2v(card), v2card(card2v(card)))

    print(0, v2card(0))
    print(Card(Suit.clubs, Rank.ten), card2v(Card(Suit.clubs, Rank.ten)))
