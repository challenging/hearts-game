import os
import sys

import glob
import copy
import time

from scipy.stats import describe

from card import read_card_games
from card import Rank, Suit, Card
from game import Game
from rules import get_setting_cards

from player import StupidPlayer, SimplePlayer
from heuristic_player import GreedyHeusisticPlayer, DynamicRankPlayer
from simulated_player import MonteCarloPlayer, MonteCarloPlayer2
from alpha_player import MCTSPlayer

# We are simulating n games accumulating a total score
nr_of_games = int(sys.argv[1])
print('We are playing {} game in total.'.format(nr_of_games))

player = SimplePlayer()
player_ai = sys.argv[2]
if player_ai.lower() == "mc2":
    player = MonteCarloPlayer2(verbose=True)
elif player_ai.lower() == "mc":
    player = MonteCarloPlayer(verbose=True)
elif player_ai.lower() == "mcts":
    player = MCTSPlayer(verbose=True)

#setting_cards = read_card_games("game/game_0008/game_1534672484.pkl")
setting_cards = get_setting_cards()
for player_idx, cards in enumerate(setting_cards[0]):
    print(player_idx, cards)

players = [player, SimplePlayer(verbose=False), SimplePlayer(), SimplePlayer(verbose=False)]

final_scores = [[], [], [], []]
for game_idx in range(nr_of_games):
    # These four players are playing the game
    game = Game(players, verbose=True)

    scores = [0, 0, 0, 0]
    for game_nr, cards in enumerate(copy.deepcopy(setting_cards)):
        #game.pass_cards()
        game._player_hands = cards
        game.play()
        game.score()
        game.reset()

        tscores = game.player_scores

        for idx, ts in enumerate(tscores):
            scores[idx] += ts

    for player_idx, (player, score) in enumerate(zip(players, scores)):
        print("{:04d} --> {:16s}({}): {:4d} points".format(game_idx+1, type(player).__name__, player_idx, score))

    for player_idx in range(4):
        final_scores[player_idx].append(scores[player_idx])

for player_idx, scores in enumerate(final_scores):
    print(player_idx, describe(scores))
