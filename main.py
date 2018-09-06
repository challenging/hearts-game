import os
import sys

import glob
import copy
import time

from random import shuffle
from scipy.stats import describe

from card import read_card_games
from card import Rank, Suit, Card
from game import Game
from rules import get_setting_cards, evaluate_players

from player import StupidPlayer, SimplePlayer
from simulated_player import MonteCarloPlayer, MonteCarloPlayer2, MonteCarloPlayer3, MonteCarloPlayer4, MonteCarloPlayer5

import mcts_player
import alpha_player


if __name__ == "__main__":
    # We are simulating n games accumulating a total score
    nr_of_games = int(sys.argv[1])
    print('We are playing {} game in total.'.format(nr_of_games))

    player = SimplePlayer()
    other_players = []

    player_ai = sys.argv[2]
    if player_ai.lower() == "mc5":
        player = MonteCarloPlayer5(verbose=True)
        other_players = [MonteCarloPlayer4(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc4":
        player = MonteCarloPlayer4(verbose=True)
        other_players = [MonteCarloPlayer3(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc3":
        player = MonteCarloPlayer3(verbose=True)
        other_players = [MonteCarloPlayer2(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc2":
        player = MonteCarloPlayer2(verbose=True)
        other_players = [MonteCarloPlayer(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc":
        player = MonteCarloPlayer(verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mcts":
        player = mcts_player.MCTSPlayer(verbose=True)
        other_players = [MonteCarloPlayer4(verbose=False) for _ in range(3)]
    else:
        player = SimplePlayer(verbose=False)
        other_players = [StupidPlayer() for _ in range(3)]

    setting_cards = read_card_games("game/game_0008/game_1534672485.pkl")
    #setting_cards = get_setting_cards()

    evaluate_players(nr_of_games, [player] + other_players, setting_cards)
