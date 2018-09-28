import os
import sys

from random import shuffle

from card import read_card_games
from card import Suit, Rank, Card
from rules import get_setting_cards, evaluate_players

from player import StupidPlayer, SimplePlayer
#from dragon_rider_player import DragonRiderPlayer
from simulated_player import MonteCarloPlayer, MonteCarloPlayer3, MonteCarloPlayer4, MonteCarloPlayer5, MonteCarloPlayer6
from new_simulated_player import MonteCarloPlayer7

#from intelligent_player import IntelligentPlayer
#from nn import PolicyValueNet


if __name__ == "__main__":
    # We are simulating n games accumulating a total score
    nr_of_games = int(sys.argv[1])
    print('We will replay {} times in total.'.format(nr_of_games))

    player = SimplePlayer()
    other_players = []

    player_ai = sys.argv[2]
    if player_ai.lower() == "mc7":
        player = MonteCarloPlayer7(verbose=True)
        debug_player = MonteCarloPlayer5(verbose=False)
        other_players = [debug_player] + [MonteCarloPlayer5(verbose=False) for player_idx in range(2)]
    elif player_ai.lower() == "mc6":
        player = MonteCarloPlayer6(verbose=True)
        other_players = [MonteCarloPlayer5(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc5":
        player = MonteCarloPlayer5(verbose=True)
        other_players = [MonteCarloPlayer4(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc4":
        player = MonteCarloPlayer4(verbose=True)
        other_players = [MonteCarloPlayer3(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc3":
        player = MonteCarloPlayer3(verbose=True)
        other_players = [MonteCarloPlayer(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "mc":
        player = MonteCarloPlayer(verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "rider":
        player = DragonRiderPlayer(self_player_idx=3, verbose=True, c_puct=1.5)
        other_players = [MonteCarloPlayer5(verbose=False) for _ in range(3)]
    elif player_ai.lower() == "intelligent":
        policy = PolicyValueNet()
        player = IntelligentPlayer(policy.policy_value, self_player_idx=3, is_selfplay=True, verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    else:
        player = SimplePlayer(verbose=False)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]

    setting_cards = read_card_games("game/game_0032/game_*.pkl")
    shuffle(setting_cards)
    evaluate_players(nr_of_games, other_players+[player], setting_cards, is_rotating=False)

    #setting_cards = get_setting_cards()
    #evaluate_players(nr_of_games, other_players+[player], setting_cards, is_rotating=False)
