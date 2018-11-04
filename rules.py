"""
This module contains a few functions comprising the rules of the game.
"""

import sys
import copy
import time

#import numpy as np
import statistics
#from scipy.stats import describe

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
            not is_broken and (card.suit != Suit.hearts or all([True if card.suit == Suit.hearts or (card.suit == Suit.spades and card.rank == Rank.queen) else False for card in hand]))
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
    card_string = [["QD,JC,JD,TD,7C,7D,5H,5C,5D,3C,KS,QC,9H",
                    "9D,9S,8C,8D,7S,6D,4C,3S,2C,2D,2S,3D,AD",
                    "JS,TS,9C,8S,6C,6S,4H,5S,4D,4S,QH,KD,TC",
                    "QS,AS,AH,KH,JH,TH,7H,6H,3H,2H,AC,KC,8H"],
                   ["TS,KH,5H,AC,KC,9C,4C,3C,2C,8D,7D,4D,2D",
                    "KS,5S,4S,9H,4H,2H,QC,TC,5C,AD,TD,9D,6D",
                    "JS,9S,8S,6S,3S,2S,7H,3H,8C,7C,6C,5D,3D",
                    "AS,QS,7S,AH,QH,JH,TH,8H,6H,JC,KD,QD,JD"],
                   ["KS,TS,8S,6S,3S,5H,3H,2H,TC,QD,6D,5D,4D",
                    "9S,5S,AH,JH,7H,6H,4H,AC,KC,JC,5C,KD,8D",
                    "4S,2S,KH,QH,8H,QC,9C,6C,4C,2C,TD,9D,7D",
                    "AS,QS,JS,7S,TH,9H,8C,7C,3C,AD,JD,3D,2D"],
                   ["QS,6S,AH,9H,QH,TH,7H,6H,3H,2H,KC,AD,8D",
                    "5S,4S,8H,5H,7C,4C,QD,TD,9D,6D,5D,4D,3D",
                    "JS,2S,4H,AC,8C,6C,5C,3C,2C,KD,JD,7D,2D",
                    "AS,KS,TS,9S,8S,7S,3S,JH,KH,QC,JC,TC,9C"],
                   ["8S,7S,3S,JC,4C,3C,9D,7D,6D,5D,4D,3D,2D",
                    "AS,KS,QS,9S,6S,4S,2S,7H,3H,AC,QD,JD,TD",
                    "TS,5S,9H,8H,6H,5H,4H,9C,8C,7C,6C,5C,2C",
                    "JS,AH,KH,QH,JH,TH,2H,KC,QC,TC,AD,KD,8D"],
                   ["QS,8S,6S,5S,2S,JH,9H,8H,2H,AC,9C,7C,3D",
                    "JS,7S,4S,3S,AH,3H,KC,QC,JC,TC,AD,TD,6D",
                    "4H,8C,6C,4C,3C,2C,JD,9D,8D,7D,5D,4D,2D",
                    "AS,KS,TS,9S,KH,QH,TH,7H,6H,5H,5C,KD,QD"]
                  ]

    card_string = [["5C, 7C, JC, QC, KC, AC, 4D, QD, 3S, 4S, 5S, 8S, 9S",
                    "2C, 3C, TC, 3D, 7D, TS, JS, KS, 3H, 5H, 7H, 8H, 9H",
                    "8C, 2D, 5D, 8D, TD, JD, AD, 2S, 6S, 7S, 2H, 6H, TH",
                    "4C, 6C, 9C, 6D, 9D, KD, QS, AS, 4H, JH, QH, KH, AH"]]


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
    sum_score = sum(scores)

    rating = [0, 0, 0, 0]

    info = zip(range(4), scores)
    pre_score, pre_rating = None, None
    for rating_idx, (player_idx, score) in enumerate(sorted(info, key=lambda x: -x[1])):
        tmp_rating = rating_idx
        if pre_score is not None:
            if score == pre_score:
                tmp_rating = pre_rating

        rating[player_idx] = tmp_rating/4 + (1-score/sum_score)

        pre_score = score
        pre_rating = tmp_rating

    return rating


def evaluate_players(nr_of_games, players, setting_cards, is_rotating=True, verbose=True):
    from game import Game

    stime = time.time()

    final_scores, proactive_moon_scores, shooting_moon_scores = [[], [], [], []], [[], [], [], []], [[], [], [], []]
    for game_idx in range(nr_of_games):
        for game_nr, cards in enumerate(copy.deepcopy(setting_cards)):
            for round_idx in range(0, 4):
                cards[0], cards[1], cards[2], cards[3] = cards[round_idx%4], cards[(round_idx+1)%4], cards[(round_idx+2)%4], cards[(round_idx+3)%4]

                for passing_direction in range(0, 4 if is_rotating else 1):
                    game = Game(players, verbose=True)

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
                        if game.players[player_idx].proactive_mode and Card(Suit.hearts, Rank.ace) in game._player_hands[player_idx]:
                            game.expose_heart_ace = True

                            for idx in range(4):
                                if player_idx != idx:
                                    game.players[idx].set_transfer_card(player_idx, Card(Suit.hearts, Rank.ace))

                        after_info.append(game._player_hands[player_idx])

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

                            print("--> {:16s}({}): {:3d} points, expose_hearts_ace={}, stats: (n={}/{}/{}, mean={:.3f}/{:.3f}/{:.3f})".format(\
                                type(player).__name__, player_idx, scores[player_idx], \
                                game.expose_heart_ace, len(final_scores[player_idx]), n_proactive_mean_moon_score, n_mean_moon_score, \
                                mean_score, proactive_mean_moon_score, mean_moon_score))

                    game.reset()
        #            break
        #        break
        #    break
        #break

    return final_scores, proactive_moon_scores, shooting_moon_scores


if __name__ == "__main__":
    scores = [208, 208, 208, 0]
    print(scores, get_rating(scores))

    scores = [0, 0, 14, 16]
    print(scores, get_rating(scores))
