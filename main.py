import os
import sys

from random import shuffle

from card import read_card_games
from card import Suit, Rank, Card
from rules import get_setting_cards, evaluate_players

from player import StupidPlayer, SimplePlayer
from simulated_player import MonteCarloPlayer, MonteCarloPlayer4, MonteCarloPlayer5
from new_simulated_player import MonteCarloPlayer7
from dragon_rider_player import RiderPlayer
from mcts_player import MCTSPlayer


if __name__ == "__main__":
    # We are simulating n games accumulating a total score
    nr_of_games = int(sys.argv[1])
    print('We will replay {} times in total.'.format(nr_of_games))

    player = SimplePlayer()
    other_players = []

    player_ai = sys.argv[2].lower()
    if player_ai == "mc7":
        player = MonteCarloPlayer7(verbose=True)
        other_players = [SimplePlayer(verbose=False) for player_idx in range(3)]
    elif player_ai == "mc5":
        player = MonteCarloPlayer5(verbose=True)
        other_players = [MonteCarloPlayer4(verbose=False) for _ in range(3)]
    elif player_ai == "mc":
        player = MonteCarloPlayer(verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    elif player_ai == "mcts":
        player = MCTSPlayer(verbose=True, c_puct=4)
        other_players = [MonteCarloPlayer7(verbose=False) for _ in range(3)]
    elif player_ai == "rider":
        player = RiderPlayer(verbose=True, c_puct=4)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    else:
        player = SimplePlayer(verbose=False)
        other_players = [StupidPlayer(verbose=False) for _ in range(3)]

    setting_cards = None
    if sys.argv[3].lower() == "small":
        setting_cards = read_card_games("game/game_0008/game_*.pkl")
        setting_cards = setting_cards[::-1]
    elif sys.argv[3].lower() == "big":
        setting_cards = read_card_games("game/game_0032/game_*.pkl")
    else:
        setting_cards = get_setting_cards()

    #shuffle(setting_cards)
    evaluate_players(nr_of_games, other_players+[player], setting_cards, is_rotating=False)
