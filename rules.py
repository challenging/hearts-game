"""
This module contains a few functions comprising the rules of the game.
"""

import sys
import copy

import numpy as np
from scipy.stats import describe

from random import shuffle, randint
from card import Suit, Rank, Card


def is_card_valid(hand, trick, card, trick_nr, is_broken):
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
        return is_broken or (
            not is_broken and (card.suit != Suit.hearts or all([card.suit == Suit.hearts for card in hand]))
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
    # shoot the moon
    card_string = [["QS,6S,AH,KH,QH,TH,7H,6H,3H,2H,KC,AD,8D",
                    "5S,4S,8H,5H,7C,4C,QD,TD,9D,6D,5D,4D,3D",
                    "AS,KS,TS,9S,8S,7S,3S,JH,9H,QC,JC,TC,9C",
                    "JS,2S,4H,AC,8C,6C,5C,3C,2C,KD,JD,7D,2D"],
                   ["JS,AH,KH,QH,JH,TH,2H,KC,QC,TC,AD,KD,8D",
                    "8S,7S,3S,JC,4C,3C,9D,7D,6D,5D,4D,3D,2D",
                    "AS,KS,QS,9S,6S,4S,2S,7H,3H,AC,QD,JD,TD",
                    "TS,5S,9H,8H,6H,5H,4H,9C,8C,7C,6C,5C,2C"],
                   ["AS,KS,TS,9S,KH,QH,TH,7H,6H,5H,5C,KD,QD",
                    "QS,8S,6S,5S,2S,JH,9H,8H,2H,AC,9C,7C,3D",
                    "JS,7S,4S,3S,AH,3H,KC,QC,JC,TC,AD,TD,6D",
                    "4H,8C,6C,4C,3C,2C,JD,9D,8D,7D,5D,4D,2D"]]

    return transform_cards(card_string)


def transform_cards(card_strings):
    cardss = []

    separator = ","
    for card_string in card_strings:
        cards = []
        for card_s in card_string:
            t = []
            for card in card_s.split(separator):
                card = card.strip()

                rank = "".join(card[:-1])
                suit = card[-1]

                t.append(transform(rank, suit))

            cards.append(t)

        cardss.append(cards)

    return cardss


def evaluate_players(nr_of_games, players, setting_cards, is_rotating=True, verbose=True, is_expose=False):
    from game import Game

    final_scores = [[], [], [], []]
    for game_idx in range(nr_of_games):
        game = Game(players, verbose=True)
        game.expose_heart_ace = is_expose

        for game_nr, cards in enumerate(copy.deepcopy(setting_cards)):
            scores = [0, 0, 0, 0]
            for round_idx in (range(0, 4 if is_rotating else 1)):
                cards_copy = copy.deepcopy(cards)

                if round_idx == 1:
                    print("pass cards to left-side")
                elif round_idx == 2:
                    print("pass cards to cross-side")
                elif round_idx == 3:
                    print("pass cards to right-side")
                else:
                    print("no passing cards")

                cards_copy[0], cards_copy[1], cards_copy[2], cards_copy[3] = \
                    cards_copy[round_idx%4], cards_copy[(round_idx+1)%4], cards_copy[(round_idx+2)%4], cards_copy[(round_idx+3)%4],

                for player_idx in range(4):
                    print("Player-{}'s init_cards: {}".format(player_idx, cards_copy[player_idx]))

                game._player_hands = cards_copy
                game.pass_cards(round_idx)

                game.play()
                game.score()

                tscores = game.player_scores

                for idx, ts in enumerate(tscores):
                    scores[idx] += ts

                if verbose:
                    for player_idx, (player, score) in enumerate(zip(players, scores)):
                        print("{:02d}:{:04d} --> {:16s}({}): {:4d} points".format(game_idx+1, round_idx, type(player).__name__, player_idx, score))

                game.reset()

            for player_idx in range(4):
                final_scores[player_idx].append(scores[player_idx])

                stats = describe(final_scores[player_idx])
                print("{:02d} --> {:16s}({}): {:4d} points, (n={}, mean={:.2f}, std={:.2f}, minmax={})".format(\
                    game_idx+1, type(game.players[player_idx]).__name__, player_idx, scores[player_idx], stats.nobs, stats.mean, stats.variance**0.5, stats.minmax))

    statss = []
    for player_idx, scores in enumerate(final_scores):
        stats = describe(scores)
        statss.append(stats)

        print("Player-{}({}): n_game, avg_score, minmax_score, std = ({}, {:.2f}, {}, {:.2f})".format(\
            player_idx, type(game.players[player_idx]).__name__, stats.nobs, stats.mean, stats.minmax, np.sqrt(stats.variance)))

    return statss


if __name__ == "__main__":
    hand = transform_cards(["8C, TC, 3S, QS, 4H, 5H, 9H, AH"])
    for card in hand[0]:
        print(card, is_card_valid(hand[0], trick=(), card=card, trick_nr=4, is_broken=False))
