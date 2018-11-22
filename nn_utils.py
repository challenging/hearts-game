#!/usr/bin/env python

from card import Suit, Rank, Card, Deck, SPADES_Q, CLUBS_T
from card import NUM_TO_INDEX, SUIT_TO_INDEX
from card import bitmask_to_card, batch_bitmask_to_card


SCORE_SCALAR = 52

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

    if isinstance(cards, list):
        for card in cards:
            results.append(card2v(card))

        for _ in range(num_slot-len(results)):
            results.append(NONE_CARD_INDEX)
    else:
        for card, prob in cards.items():
            results.append(prob)

        for _ in range(num_slot-len(results)):
            results.append(0.0)

    return results


def transform_game_info_to_nn(state, trick_nr):
    a_memory = []

    remaining_cards = []
    for suit, info in enumerate(state.current_info):
        remaining_cards.extend(list(batch_bitmask_to_card(suit, info[1])))
    a_memory.append(remaining_cards)

    current_trick_nr = trick_nr+len(state.tricks)-1
    a_memory.append(current_trick_nr)

    leading_idx = (state.start_pos+3-len(state.tricks[-1][1])+1)%4
    trick_order = [(leading_idx+idx)%4 for idx in range(4)]
    a_memory.append(trick_order)

    pos = state.position
    a_memory.append(pos)

    played_order = (current_trick_nr-1)*4+len(state.tricks[-1][1])
    a_memory.append(played_order)

    trick_cards = [bitmask_to_card(suit, rank) for suit, rank in state.tricks[-1][1]]
    a_memory.append(trick_cards)

    must_cards = [[], [], [], []]
    for player_idx in range(4):
        must_cards[player_idx] = state.must_have.get(player_idx, [])
    a_memory.append(must_cards)

    historical_cards = [[], [], [], []]
    for player_idx in range(4):
        historical_cards[player_idx] = [bitmask_to_card(suit, rank) for suit, rank in state.historical_cards[player_idx]]
    a_memory.append(historical_cards)

    score_cards = [[], [], [], []]
    for player_idx, cards in enumerate(state.score_cards):
        for suit, ranks in enumerate(cards):
            if suit == SUIT_TO_INDEX["C"] and ranks & NUM_TO_INDEX["T"]:
                score_cards[player_idx].append(CLUBS_T)
            elif suit == SUIT_TO_INDEX["S"] and ranks & NUM_TO_INDEX["Q"]:
                score_cards[player_idx].append(SPADES_Q)
            elif suit == SUIT_TO_INDEX["H"]:
                score_cards[player_idx].extend(list(batch_bitmask_to_card(suit, ranks)))
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
    for suit, ranks in state.get_valid_cards(state.hand_cards[state.start_pos], current_trick_nr, is_playout=True).items():
        bitmask = NUM_TO_INDEX["2"]
        while bitmask <= NUM_TO_INDEX["A"]:
            if ranks & bitmask:
                valid_cards.append(bitmask_to_card(suit, bitmask))

            bitmask <<= 1
    a_memory.append(valid_cards)

    expose_info = state.expose_info
    a_memory.append(expose_info)

    void_info = [[], [], [], []]
    for player_idx in range(4):
        for suit, is_void in sorted(state.void_info[player_idx].items(), key=lambda x: x[0]):
            void_info[player_idx].append(is_void)
    a_memory.append(void_info)

    winning_info = state.winning_info
    a_memory.append(winning_info)

    #print_a_memory(a_memory)

    return full_cards(remaining_cards), \
           current_trick_nr-1, trick_order, pos, played_order, limit_cards(trick_cards, 3), \
           [limit_cards(must_cards[0], 4), limit_cards(must_cards[1], 4), limit_cards(must_cards[2], 4), limit_cards(must_cards[3], 4)], \
           [limit_cards(historical_cards[0], 13), limit_cards(historical_cards[1], 13), limit_cards(historical_cards[2], 13), limit_cards(historical_cards[3], 13)], \
           [limit_cards(score_cards[0], 15), limit_cards(score_cards[1], 15), limit_cards(score_cards[2], 15), limit_cards(score_cards[3], 15)], \
           limit_cards(hand_cards, 13), limit_cards(valid_cards, 13), \
           expose_info, void_info, winning_info


def print_a_memory(played_data):
    remaining_cards = played_data[0]
    trick_nr = played_data[1]
    trick_order = played_data[2]
    trick_pos = played_data[3]
    played_order = played_data[4]
    trick_cards = played_data[5]
    must_cards = played_data[6]
    historical_cards = played_data[7]
    score_cards = played_data[8]
    hand_cards = played_data[9]
    valid_cards = played_data[10]
    expose_info = played_data[11]
    void_info = played_data[12]
    winning_info = played_data[13]
    probs = played_data[14] if len(played_data) > 14 else None
    score = played_data[15] if len(played_data) > 15 else None

    print("remaining_cards:", [(card, card2v(card)) for card in remaining_cards])
    print("remaining_cards:", full_cards(remaining_cards))

    print("       trick_nr:", trick_nr)
    print("    trick_order:", trick_order)
    print("      trick_pos:", trick_pos)
    print("   played_order:", played_order)
    print("    trick_cards:", [(card, card2v(card)) for card in trick_cards])
    print("    trick_cards:", limit_cards(trick_cards, 3))

    print("     must_cards:", [[(card, card2v(card)) for card in cards] for cards in must_cards])
    print("     must_cards:", [limit_cards(cards, 4) for cards in must_cards])

    print("historical_card:", [[(card, card2v(card)) for card in cards] for cards in historical_cards])
    print("historical_card:", [limit_cards(cards, 13) for cards in historical_cards])

    print("    score_cards:", [[(card, card2v(card)) for card in cards] for cards in score_cards])
    print("    score_cards:", [limit_cards(cards, 15) for cards in score_cards])

    print("     hand_cards:", hand_cards)
    print("     hand_cards:", limit_cards(hand_cards, 13))

    print("    valid_cards:", valid_cards)
    print("    valid_cards:", limit_cards(valid_cards, 13))

    print("    expose_info:", expose_info)
    print("      void_info:", void_info)
    print("   winning_info:", winning_info)

    if probs:
        print("      probs:", limit_cards(dict(zip(valid_cards, probs)), 13))
        print("      score:", score)

    print()


if __name__ == "__main__":
    for card in Deck().cards:
        print(card, card2v(card), v2card(card2v(card)))

    print(0, v2card(0))
    print(Card(Suit.clubs, Rank.ten), card2v(Card(Suit.clubs, Rank.ten)))
