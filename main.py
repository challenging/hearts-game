import os
import sys

from random import shuffle

from card import read_card_games
from card import Suit, Rank, Card
from rules import get_setting_cards, evaluate_players

from player import StupidPlayer, SimplePlayer
from simulated_player import MonteCarloPlayer, MonteCarloPlayer3
from simulated_player import MonteCarloPlayer4, MonteCarloPlayer5, MonteCarloPlayer6
from new_simulated_player import MonteCarloPlayer7
from new_simple_player import NewSimplePlayer
from dragon_rider_player import DragonRiderPlayer

#from intelligent_player import IntelligentPlayer
#from nn import PolicyValueNet


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
    elif player_ai == "mc6":
        player = MonteCarloPlayer6(verbose=True)
        other_players = [MonteCarloPlayer5(verbose=False) for _ in range(3)]
    elif player_ai == "mc5":
        player = MonteCarloPlayer5(verbose=True)
        other_players = [MonteCarloPlayer4(verbose=False) for _ in range(3)]
    elif player_ai == "mc4":
        player = MonteCarloPlayer4(verbose=True)
        other_players = [MonteCarloPlayer3(verbose=False) for _ in range(3)]
    elif player_ai == "mc3":
        player = MonteCarloPlayer3(verbose=True)
        other_players = [MonteCarloPlayer(verbose=False) for _ in range(3)]
    elif player_ai == "mc":
        player = MonteCarloPlayer(verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    elif player_ai == "rider":
        player = DragonRiderPlayer(self_player_idx=3, verbose=True, c_puct=1.5)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    elif player_ai == "intelligent":
        policy = PolicyValueNet()
        player = IntelligentPlayer(policy.policy_value, self_player_idx=3, is_selfplay=True, verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    elif player_ai == "new_simple":
        player = NewSimplePlayer(verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    else:
        player = SimplePlayer(verbose=False)
        other_players = [StupidPlayer(verbose=False) for _ in range(3)]

    setting_cards = None
    if sys.argv[3].lower() == "small":
        setting_cards = read_card_games("game/game_0008/game_*.pkl")
    elif sys.argv[3].lower() == "big":
        setting_cards = read_card_games("game/game_0032/game_*.pkl")
    else:
        setting_cards = get_setting_cards()

    #shuffle(setting_cards)
    evaluate_players(nr_of_games, other_players+[player], setting_cards, is_rotating=False)
