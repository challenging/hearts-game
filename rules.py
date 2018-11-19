"""
This module contains a few functions comprising the rules of the game.
"""

import sys
import copy
import time

import statistics

from random import shuffle, randint

from card import Suit, Rank, Card
from card import SPADES_Q, CLUBS_T


def is_card_valid(hand, trick, card, trick_nr, is_broken):
    if trick_nr == 0 and len(trick) == 0:
        if card.suit == Suit.clubs and card.rank == Rank.two:
            return True
        else:
            return False

    if trick_nr == 0 and card_points(card) > 0:
        return False

    if not trick:
        return is_broken or (
            not is_broken and \
            (card.suit != Suit.hearts or \
             all([True if card.suit == Suit.hearts or (card.suit == Suit.spades and card.rank == Rank.queen) else False for card in hand]))
        )

    leading_suit = trick[0].suit

    return card.suit == leading_suit or all([card.suit != leading_suit for card in hand])


def is_score_card(card):
    return card.suit == Suit.hearts or card in [CLUBS_T, SPADES_Q]


def card_points(card):
    score = 1 if card.suit == Suit.hearts else 0 + (13 if card == SPADES_Q else 0)

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
    card_string = [["5C, 7C, JC, QC, KC, AC, 4D, QD, 3S, 4S, 5S, 8S, 9S",
                    "2C, 3C, TC, 3D, 7D, TS, JS, KS, 3H, 5H, 7H, 8H, 9H",
                    "8C, 2D, 5D, 8D, TD, JD, AD, 2S, 6S, 7S, 2H, 6H, TH",
                    "4C, 6C, 9C, 6D, 9D, KD, QS, AS, 4H, JH, QH, KH, AH"]]

    card_string = [["5C, 7C, JC, QC, KC, AC, 4D, QD, 3S, 4S, 5S, 8S, 9S",
                    "4C, 6C, 9C, 6D, 9D, KD, QS, AS, 4H, JH, QH, KH, AH",
                    "8C, 2D, 5D, 8D, TD, JD, AD, 2S, 6S, 7S, 2H, 6H, TH",
                    "2C, 3C, TC, 3D, 7D, TS, JS, KS, 3H, 5H, 7H, 8H, 9H"]]

    return transform_cards(card_string)


def transform_cards(card_strings):
    cardss = []

    separator = ','
    for card_string in card_strings:
        cards = []
        for card_s in card_string:
            t = []
            for card in card_s.split(separator):
                card = card.strip()

                rank = card[0]#"".join(card[:-1])
                suit = card[1]

                t.append(transform(rank, suit))

            cards.append(t)

        cardss.append(cards)

    return cardss


def get_rating(scores):
    return [-score for score in scores]

def evaluate_players(nr_of_games, players, setting_cards, is_rotating=True, verbose=True, out_file=sys.stdout):
    from game import Game

    stime = time.time()

    final_scores, proactive_moon_scores, shooting_moon_scores = [[], [], [], []], [[], [], [], []], [[], [], [], []]
    for game_idx in range(nr_of_games):
        for game_nr, cards in enumerate(copy.deepcopy(setting_cards)):
            for round_idx in range(0, 4):
                cards[0], cards[1], cards[2], cards[3] = cards[round_idx%4], cards[(round_idx+1)%4], cards[(round_idx+2)%4], cards[(round_idx+3)%4]

                for passing_direction in range(0, 4 if is_rotating else 1):
                    game = Game(players, verbose=True, out_file=out_file)

                    game._player_hands = copy.deepcopy(cards)

                    """
                    if passing_direction == 0:
                        print("pass cards to left-side")
                    elif passing_direction == 1:
                        print("pass cards to cross-side")
                    elif passing_direction == 2:
                        print("pass cards to right-side")
                    else:
                        print("no passing cards")
                    """

                    before_info = []
                    for player_idx in range(4):
                        before_info.append(cards[player_idx])

                    game.pass_cards(passing_direction)

                    after_info = []
                    for player_idx in range(4):
                        after_info.append(game._player_hands[player_idx])

                    if not game.expose_heart_ace:
                        for player_idx in range(4):
                            is_exposed = players[player_idx].expose_hearts_ace(game._player_hands[player_idx])
                            game.expose_heart_ace = is_exposed

                            if game.expose_heart_ace:
                                out_file.write("Player-{} exposes HEARTS ACE\n".format(player_idx))

                    for player_idx, (before_cards, after_cards) in enumerate(zip(before_info, after_info)):
                        out_file.write("Player-{}'s init_cards: {} to {}\n".format(player_idx, before_cards, after_cards))

                    for player_idx in range(4):
                        if hasattr(game, "evaluate_proactive_mode"):
                            game.players[player_idx].evaluate_proactive_mode(game._player_hands[player_idx])

                    game.play()
                    game.score()

                    scores = [0, 0, 0, 0]
                    if game.is_shootmoon:
                        is_proactive_mode = any([True if player.proactive_mode else False for player in game.players])

                        if is_proactive_mode:
                            for player_idx, score in enumerate(game.player_scores):
                                proactive_moon_scores[player_idx].append(score)

                                scores[player_idx] = score
                        else:
                            for player_idx, score in enumerate(game.player_scores):
                                shooting_moon_scores[player_idx].append(score)

                                scores[player_idx] = score
                    else:
                        for player_idx, score in enumerate(game.player_scores):
                            final_scores[player_idx].append(score)

                            scores[player_idx] = score

                    if verbose and final_scores[player_idx]:
                        for player_idx, player in enumerate(players):
                            mean_score = statistics.mean(final_scores[player_idx])

                            n_proactive_mean_moon_score = sum([1 for score in proactive_moon_scores[player_idx] if score == 0])
                            proactive_mean_moon_score = statistics.mean(proactive_moon_scores[player_idx]+final_scores[player_idx])

                            n_mean_moon_score = sum([1 for score in shooting_moon_scores[player_idx] if score == 0])
                            mean_moon_score = statistics.mean(shooting_moon_scores[player_idx]+proactive_moon_scores[player_idx]+final_scores[player_idx])

                            out_file.write("--> {:16s}({}): {:3d} points, expose_hearts_ace={}, stats: (n={}/{}/{}, mean={:.3f}/{:.3f}/{:.3f})\n".format(\
                                type(player).__name__, player_idx, scores[player_idx], \
                                game.expose_heart_ace, len(final_scores[player_idx]), n_proactive_mean_moon_score, n_mean_moon_score, \
                                mean_score, proactive_mean_moon_score, mean_moon_score))

                    game.reset()

                    if len(setting_cards) == 1: break
                if len(setting_cards) == 1 and nr_of_games == 1: break
            if len(setting_cards) == 1 and nr_of_games == 1: break
        if len(setting_cards) == 1 and nr_of_games == 1: break

    return final_scores, proactive_moon_scores, shooting_moon_scores
