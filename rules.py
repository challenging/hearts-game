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
    card_string = [["4C, KC, 2D, 5D, 8D, 9D, QD, 4S, 7S, 3H, 8H, TH, AH",
                    "2C, 3C, 5C, 9C, TC, 6D, AD, 3S, QS, AS, 9H, JH, KH",
                    "6C, 7C, JC, QC, AC, 2S, 5S, 6S, 9S, KS, 6H, 7H, QH",
                    "8C, 3D, 4D, 7D, TD, JD, KD, 8S, TS, JS, 2H, 4H, 5H"]]

    card_string = [["6C, 7C, JC, QC, AC, 2S, 5S, 6S, 9S, KS, 6H, 7H, QH",
                    "8C, 3D, 4D, 7D, TD, JD, KD, 8S, TS, JS, 2H, 4H, 5H",
                    "4C, KC, 2D, 5D, 8D, 9D, QD, 4S, 7S, 3H, 8H, TH, AH",
                    "2C, 3C, 5C, 9C, TC, 6D, AD, 3S, QS, AS, 9H, JH, KH"]]

    card_string = [["2C, 8C, 9C, JC, 3D, 9D, 4S, 5S, QS, KS, 8H, 9H, AH",
                    "QC, 8D, AD, 2S, 3S, 6S, 7S, 9S, 2H, 3H, 5H, TH, KH",
                    "6C, TC, 4D, 6D, 7D, TD, KD, 8S, TS, 4H, 6H, 7H, JH",
                    "3C, 4C, 5C, 7C, KC, AC, 2D, 5D, JD, QD, JS, AS, QH"]]

    card_string = [["6C, 7C, KC, 2D, 8D, 2S, 5S, 7S, 9S, AS, 7H, JH, KH",
                    "4C, 5C, 9C, TC, 4D, QD, KD, AD, QS, 2H, 5H, 9H, AH",
                    "QC, AC, 5D, 9D, 3S, TS, JS, KS, 3H, 6H, 8H, TH, QH",
                    "2C, 3C, 8C, JC, 3D, 6D, 7D, TD, JD, 4S, 6S, 8S, 4H"]]

    card_string = [["4C, 3D, 9D, QD, AD, 4S, 5S, 8S, KS, 4H, 5H, JH, KH",
                    "3C, 5C, 7C, 9C, JC, QC, KD, 2S, 3S, 9S, JS, 7H, 9H",
                    "8C, AC, 2D, 4D, 6D, 7D, TD, JD, 7S, AS, 6H, 8H, AH",
                    "2C, 6C, TC, KC, 5D, 8D, 6S, TS, QS, 2H, 3H, TH, QH"]]

    card_string = [["2C, 4C, 6C, JC, 2S, 4S, 5S, 9S, 3H, 4H, 8H, 9H, JH",
                    "7C, 8C, TC, KC, AC, 8D, JD, AD, 6S, TS, JS, AS, AH",
                    "3C, 5C, 9C, QC, 7D, 9D, QD, KD, 3S, 8S, 6H, TH, QH",
                    "2D, 3D, 4D, 5D, 6D, TD, 7S, QS, KS, 2H, 5H, 7H, KH"]]

    card_string = [["2C, TC, AC, 2D, 8D, 9D, QD, AD, 7S, 9S, TS, 9H, JH",
                    "3C, 4C, 7D, 6S, QS, KS, AS, 3H, 6H, 7H, TH, QH, AH",
                    "5C, 7C, 9C, QC, KC, 3D, 5D, 6D, 2S, 4S, 5S, 8S, KH",
                    "6C, 8C, JC, 4D, TD, JD, KD, 3S, JS, 2H, 4H, 5H, 8H"]]

    card_string = [["2C, 3C, 4C, 6C, 3D, 6D, 7D, 8D, 4S, QS, 8H, 9H, QH",
                    "7C, 9C, KC, AC, 2D, 5D, TD, 2S, 8S, KS, AS, 3H, KH",
                    "5C, 8C, QC, 4D, KD, 3S, 6S, 7S, 9S, JS, 2H, 4H, AH",
                    "TC, JC, 9D, JD, QD, AD, 5S, TS, 5H, 6H, 7H, TH, JH"]]

    card_string = [["4C, 5C, 9C, TC, KC, AC, 6D, JD, 4S, 5S, 7S, 9S, QS",
                    "2C, 7C, 2D, 3D, 5D, 7D, 9D, KD, JS, 6H, 7H, 8H, KH",
                    "3C, 6C, 8C, QC, 8D, TD, QD, 2S, TS, KS, 3H, 4H, JH",
                    "JC, 4D, AD, 3S, 6S, 8S, AS, 2H, 5H, 9H, TH, QH, AH"]]

    card_string = [["2C, 7C, 2D, 3D, 5D, 7D, 9D, KD, JS, 6H, 7H, 8H, KH",
                    "3C, 6C, 8C, QC, 8D, TD, QD, 2S, TS, KS, 3H, 4H, JH",
                    "JC, 4D, AD, 3S, 6S, 8S, AS, 2H, 5H, 9H, TH, QH, AH",
                    "4C, 5C, 9C, TC, KC, AC, 6D, JD, 4S, 5S, 7S, 9S, QS"]]

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

                    if passing_direction == 0:
                        print("pass cards to left-side")
                    elif passing_direction == 1:
                        print("pass cards to cross-side")
                    elif passing_direction == 2:
                        print("pass cards to right-side")
                    else:
                        print("no passing cards")

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
                                print("Player-{} exposes HEARTS ACE".format(player_idx))

                    for player_idx, (before_cards, after_cards) in enumerate(zip(before_info, after_info)):
                        print("Player-{}'s init_cards: {} to {}".format(player_idx, before_cards, after_cards))

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
                    if len(setting_cards) == 1 and nr_of_games == 1: break
                if len(setting_cards) == 1 and nr_of_games == 1: break
            if len(setting_cards) == 1 and nr_of_games == 1: break
        if len(setting_cards) == 1 and nr_of_games == 1: break

    return final_scores, proactive_moon_scores, shooting_moon_scores
