#!/usr/bin/env python

from card import Suit, Rank, Card, Deck


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



if __name__ == "__main__":
    for card in Deck().cards:
        print(card, card2v(card), v2card(card2v(card)))

    print(0, v2card(0))
    print(Card(Suit.clubs, Rank.ten), card2v(Card(Suit.clubs, Rank.ten)))
