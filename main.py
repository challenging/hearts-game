import os
import sys

from random import shuffle

from card import read_card_games
from card import Suit, Rank, Card
from rules import get_setting_cards, evaluate_players

from player import StupidPlayer, SimplePlayer
from dragon_rider_player import DragonRiderPlayer
from simulated_player import MonteCarloPlayer, MonteCarloPlayer2, MonteCarloPlayer3, MonteCarloPlayer4, MonteCarloPlayer5, MonteCarloPlayer6
from simulated_player import MonteCarloPlayer33

import mcts_player
import alpha_player


if __name__ == "__main__":
    # We are simulating n games accumulating a total score
    nr_of_games = int(sys.argv[1])
    print('We will replay {} times in total.'.format(nr_of_games))

    player = SimplePlayer()
    other_players = []

    player_ai = sys.argv[2]
    if player_ai.lower() == "mc6":
        player = MonteCarloPlayer6(verbose=True)
        other_players = [MonteCarloPlayer5(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc5":
        player = MonteCarloPlayer5(verbose=True)
        other_players = [MonteCarloPlayer4(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc4":
        player = MonteCarloPlayer4(verbose=True)
        other_players = [MonteCarloPlayer3(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc3":
        player = MonteCarloPlayer33(verbose=True)
        other_players = [MonteCarloPlayer3(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc2":
        player = MonteCarloPlayer2(verbose=True)
        other_players = [MonteCarloPlayer(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc":
        player = MonteCarloPlayer(verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mcts":
        player = mcts_player.MCTSPlayer(C=72, verbose=True)   # 48
        other_players = [MonteCarloPlayer5(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "rider":
        player = DragonRiderPlayer(self_player_idx=3, verbose=True, c_puct=2)
        other_players = [MonteCarloPlayer6(verbose=False) for _ in range(3)]
    else:
        player = SimplePlayer(verbose=False)
        other_players = [StupidPlayer() for _ in range(3)]

    #setting_cards = read_card_games("game/game_0008/game_1534672482.pkl")
    #setting_cards = read_card_games("game/game_0032/game_1536341876.pkl")
    setting_cards = read_card_games("game/game_0032/game_*.pkl")
    #setting_cards = read_card_games("game/game_0008/game_153467248*.pkl")

    shuffle(setting_cards)
    evaluate_players(nr_of_games, other_players+[player], setting_cards, is_rotating=False)

    #setting_cards = get_setting_cards()
    #evaluate_players(nr_of_games, other_players+[player], setting_cards, is_rotating=True)
