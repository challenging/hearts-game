"""
This module contains a few functions comprising the rules of the game.
"""

import sys
import copy

from random import shuffle, randint
from card import Suit, Rank, Card

def is_card_valid(hand, trick, card, trick_nr, are_hearts_broken):
    """
    Return True if the given card is valid to play in given context, False otherwise.
    """

    if trick_nr == 0 and len(trick) == 0:
        if card.suit == Suit.clubs and card.rank == Rank.two:
            return True
        else:
            return False

    # No points allowed in first trick
    if trick_nr == 0 and card_points(card) > 0:
        return False

    # No hearts can be led until hearts are broken
    if not trick:
        return are_hearts_broken or (
            not are_hearts_broken and (card.suit != Suit.hearts
                                       or all([card.suit == Suit.hearts for card in hand]))
        )

    # Suit must be followed unless player has none of that suit
    leading_suit = trick[0].suit

    return card.suit == leading_suit or all([card.suit != leading_suit for card in hand])


def is_score_card(card):
    if card.suit == Suit.hearts or card in [Card(Suit.clubs, Rank.ten), Card(Suit.spades, Rank.queen)]:
        return True
    else:
        return False


def card_points(card):
    """
    Return the number of points given card is worth.
    """

    score = 1 if card.suit == Suit.hearts else 0 + (13 if card == Card(Suit.spades, Rank.queen) else 0)

    return score


def reversed_score(cards):
    t = []
    for card in cards:
        if card.suit == Suit.hearts or (card.suit == Suit.spades and card.rank == Rank.queen):
            t.append(card)

    return len(t) == 14


def transform(rank, suit):
    if suit == "S":
        suit = Suit.spades
    elif suit == "H":
        suit = Suit.hearts
    elif suit == "C":
        suit = Suit.clubs
    elif suit == "D":
        suit = Suit.diamonds

    if rank == "A":
        rank = Rank.ace
    elif rank == "2":
        rank = Rank.two
    elif rank == "3":
        rank = Rank.three
    elif rank == "4":
        rank = Rank.four
    elif rank == "5":
        rank = Rank.five
    elif rank == "6":
        rank = Rank.six
    elif rank == "7":
        rank = Rank.seven
    elif rank == "8":
        rank = Rank.eight
    elif rank == "9":
        rank = Rank.nine
    elif rank == "10" or rank == "T":
        rank = Rank.ten
    elif rank == "J":
        rank = Rank.jack
    elif rank == "Q":
        rank = Rank.queen
    elif rank == "K":
        rank = Rank.king

    return Card(suit, rank)


def get_setting_cards():
    card_string = ["JS, JH, 9C, 9D, 7S, 6H, 4D, 3D, 3S, 2S, AS, QS, KC",
                   "KS, 10S, QH, JC, 10D, 9H, 7D, 7H, 6C, 4H, 3C, 2D, 2H",
                   "AC, AH, QD, 10H, 8C, 8S, 7C, 6D, 6S, 4C, 4S, 3H, 2C",
                   "KD, KH, QC, JD, AD, 10C, 9S, 8D, 8H, 5C, 5D, 5S, 5H"]

    card_string = ["4C, 9C, 2D, 4D, 5D, 9D, 5S, QS, 2H, 4H, 9H, QH, JC",
                   "3C, 6D, 8D, 10D, 10S, QD, KS, AC, AD, 5H, 8H, JH, 2C",
                   "AS, KC, KH, JS, 10C, 7C, 7D, 7H, 4S, 3D, 3S, 3H, 5C",
                   "AH, KD, JD, 10H, 9S, 8C, 8S, 7S, 6C, 6S, 6H, 2S, QC"]

    card_string = ["QC, 3D, 4D, 5D, 8D, 9D, JD, AD, 5S, JS, KS, 10H, 10C",
                   "2S, 4S, 7D, 9C, 9S, JC, QS, KD, 3H, 6H, 8H, JH, 2C",
                   "AS, QD, QH, 10D, 9H, 8C, 8S, 6C, 5H, 4C, 4H, 2D, 3C",
                   "AH, KC, KH, 10S, 7C, 7S, 7H, 6D, 6S, 5C, 3S, 2H, AC"]

    card_string = ["4C, 3D, 9D, QD, KD, AD, KS, AS, 2H, 5H, QH, KH, 5C",
                   "JC, JS, 10C, 10S, 10H, 9C, 8H, 6C, 6S, 5D, 4H, 3S, QC",
                   "AC, KC, JD, JH, 10D, 9S, 8D, 7D, 6D, 4D, 3H, 2D, 3C",
                   "7C, 8C, 2S, 4S, 5S, 7S, 8S, QS, 6H, 7H, 9H, AH, 2C"]

    card_string = ["6C, KD, 9S, 10S, JS, QS, AS, 4H, 8H, 9H, JH, QH, 2C",
                   "KS, AH, KH, QD, JD, 9C, 6D, 5H, 4C, 4S, 3C, 3D, 10C",
                   "AD, 10H, 9D, 8D, 7S, 5C, 5D, 5S, 4D, 3H, 2S, 2H, 8C",
                   "AC, KC, QC, JC, 10D, 8S, 7D, 7H, 6S, 6H, 3S, 2D, 7C"]

    card_string = ["8C, 10C, AC, 4D, 3S, 4S, 9S, QS, 4H, 5H, 9H, AH, JC",
                   "QC, JH, 10D, 10H, 8D, 6D, 6S, 6H, 5D, 3C, 2D, 2S, 2C",
                   "AS, KS, QD, JS, 10S, 9D, 8S, 7D, 5C, 5S, 4C, 2H, KC",
                   "AD, KD, KH, QH, JD, 9C, 8H, 7C, 7S, 7H, 3D, 3H, 6C"]

    card_string = ["10C, JC, 9D, 3S, 6S, JS, KS, 8H, JH, QH, KH, AH, 5C",
                   "3C, 6C, 8C, QC, KC, 5D, 7D, 4S, 8S, 9S, QS, 6H, 4C",
                   "AC, AD, KD, QD, JD, 10S, 7H, 6D, 4D, 3H, 2D, 2S, 2C",
                   "7C, 3D, 8D, 10D, 5S, 7S, AS, 2H, 4H, 5H, 9H, 10H, 9C"]

    card_string = ["AC, 5D, 8D, 5S, 6S, 8S, 9S, QS, AS, 8H, 9H, JH, QC",
                   "KH, 10C, 10H, 9D, 7D, 7H, 6C, 5C, 4C, 4H, 3C, 2S, 2C",
                   "KS, AD, AH, QD, 7S, 6H, 4D, 4S, 3D, 3S, 2D, 2H, KC",
                   "7C, 8C, JC, 6D, 10D, JD, KD, 10S, JS, 3H, 5H, QH, 9C"]
    """
    the winning_player_index is 2
    player 2(MonteCarloPlayer) win this  1 trick by 8C card based on [2C, 8C, 5C, 7C]
    after   1 round, status of every players' hand cards
    ==================================================================
    0 MonteCarloPlayer2 [6C, QC, KC, 3D, 8D, 10D, QD, KD, 4S, QS, 8H, AH] 0
    1 SimplePlayer [3C, 4C, 10C, JC, AC, 4D, AD, 6S, 2H, 4H, 7H, 10H] 0
    2 MonteCarloPlayer [2S, 5S, 7S, 8S, 10S, JS, KS, AS, 3H, 5H, 6H, JH] 0
    3 SimplePlayer [KH, QH, JD, 9C, 9D, 9S, 9H, 7D, 6D, 5D, 3S, 2D] 0
    """

    """
    player 1(SimplePlayer) win this  1 trick by JC card based on [2C, JC, 10C, 7C]
    after   1 round, status of every players' hand cards
    ==================================================================
    0 MCTSPlayer [6C, 8C, 9C, 3D, 4D, QD, 2S, 7S, 10S, QS, 10H, JH] 0
    1 SimplePlayer [KS, AD, KD, 9D, 9S, 9H, 8S, 6D, 5D, 4H, 3S, 2H] 0
    2 SimplePlayer [AH, QC, QH, JD, 8D, 7H, 6H, 5C, 5S, 5H, 3C, 3H] 0
    3 SimplePlayer [AS, AC, KC, KH, JS, 10D, 8H, 7D, 6S, 4C, 4S, 2D] 0

    Player 1(SimplePlayer) played 3S card as the leading card
    (count_simulation, wins, plays, percent_wins, played_card, valid_cards) = (116, 113, 113, 0.325, 2S, [2S, 7S, 10S, QS])

    """

    card_string = ["6C, 8C, 9C, 3D, 4D, QD, 2S, 7S, 10S, QS, 10H, JH, 2C",
                   "KS, AD, KD, 9D, 9S, 9H, 8S, 6D, 5D, 4H, 3S, 2H, JC",
                   "AH, QC, QH, JD, 8D, 7H, 6H, 5C, 5S, 5H, 3C, 3H, 10C",
                   "AS, AC, KC, KH, JS, 10D, 8H, 7D, 6S, 4C, 4S, 2D, 7C"]

    """
    player 2(StupidPlayer) win this  1 trick by 9C card based on [2C, 9C, 8C, 5C]
    after   1 round, status of every players' hand cards
    ==================================================================
    0 MCTSPlayer [JC, KC, 6D, 8D, 9D, JD, QD, KD, 3S, JS, AS, 9H] 0
    1 SimplePlayer [4C, 6C, 7C, QC, 2S, TS, QS, KS, 4H, 8H, JH, QH] 0
    2 StupidPlayer [TH, 8S, 4S, KH, 9S, 7D, 2D, 5S, TD, 5H, AH, 4D] 0
    3 SimplePlayer [AC, AD, TC, 7S, 7H, 6S, 6H, 5D, 3C, 3D, 3H, 2H] 0
    """

    card_string = ["JC, KC, 6D, 8D, 9D, JD, QD, KD, 3S, JS, AS, 9H, 5C",
                   "4C, 6C, 7C, QC, 2S, TS, QS, KS, 4H, 8H, JH, QH, 2C",
                   "TH, 8S, 4S, KH, 9S, 7D, 2D, 5S, TD, 5H, AH, 4D, 9C",
                   "AC, AD, TC, 7S, 7H, 6S, 6H, 5D, 3C, 3D, 3H, 2H, 8C"]

    return [transform_cards(card_string)]


def transform_cards(card_string):
    cards = []
    for card_s in card_string:
        t = []
        for card in card_s.split(","):
            card = card.strip()

            rank = "".join(card[:-1])
            suit = card[-1]

            t.append(transform(rank, suit))

        cards.append(t)

    return cards


def redistribute_card(copy_cards, info):
    remaining_cards = [c for card in copy_cards for c in card]

    ori_size = [len(card) for card in copy_cards]
    new_cards = [[], [], [], []]

    player_idxs = []
    for player_idx, lacking_info in enumerate(info):
        player_idxs.append((player_idx, sum(lacking_info.values())))

    while sum([player_idx for player_idx in range(4) if len(new_cards[player_idx]) != len(copy_cards[player_idx])]):
        cards = copy.deepcopy(copy_cards)
        shuffle(remaining_cards)

        new_cards = [[], [], [], []]
        removed_cards = []
        for card in remaining_cards:
            for player_idx, _ in sorted(player_idxs, key=lambda x: -x[1]):
                if info[player_idx][card.suit] == False and len(new_cards[player_idx]) < ori_size[player_idx]:
                    new_cards[player_idx].append(card)
                    removed_cards.append(card)

                    break

        lacking_player_idxs = [player_idx for player_idx in range(4) if len(new_cards[player_idx]) < len(cards[player_idx])]
        print("enter re-redistribute cards", new_cards, cards, lacking_player_idxs, [len(new_cards[player_idx]) == len(cards[player_idx]) for player_idx in range(4)])

        is_found = False
        for lacking_player_idx in lacking_player_idxs:
            for card in set(remaining_cards) - set(removed_cards):
                for player_idx, hand_cards in enumerate(new_cards):
                    if lacking_player_idx != player_idx and info[player_idx][card.suit] == False:
                        for given_card in new_cards[player_idx]:
                            if info[lacking_player_idx][given_card.suit] == False:
                                print("{}'s {} to {}, and get {}".format(player_idx, given_card, lacking_player_idx, card))

                                new_cards[lacking_player_idx].append(given_card)
                                new_cards[player_idx].remove(given_card)
                                new_cards[player_idx].append(card)

                                is_found = True

                                break
                        if is_found:
                            break
                if is_found:
                    break
            if is_found:
               break

    print(cards, new_cards)


if __name__ == "__main__":
    #print(get_setting_cards())

    """
    cards = ["8C, TC, 3S, QS, 4H, 5H, 9H, AH",
             "QC, JH, TD, TH, 8D, 6D, 6H, 5D",
             "AS, KS, KC, QD, JS, TS, 9D, 2H",
             "KD, KH, QH, JD, 8H, 7H, 6C, 3H"]
    """
    #cards = ["JH", "4C", "6D", "8D"]
    cards = ["4C,7H", "AD", "JH", "4S,7S"]
    cards = transform_cards(cards)

    info = [{Suit.spades: True, Suit.hearts: False, Suit.diamonds: True, Suit.clubs: False},
            {Suit.spades: False, Suit.hearts: False, Suit.diamonds: False, Suit.clubs: False},
            {Suit.spades: False, Suit.hearts: False, Suit.diamonds: True, Suit.clubs: True},
            {Suit.spades: False, Suit.hearts: True, Suit.diamonds: True, Suit.clubs: False}]

    for _ in range(32):
        redistribute_card(cards, info)
