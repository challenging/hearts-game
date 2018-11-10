import os
import sys

from random import shuffle

from card import read_card_games
from card import Suit, Rank, Card
from rules import get_setting_cards, evaluate_players

from player import StupidPlayer, SimplePlayer
from simulated_player import MonteCarloPlayer
from new_simple_player import NewSimplePlayer
from new_simulated_player import MonteCarloPlayer7
from dragon_rider_player import RiderPlayer

from mcts import policy_value_fn


if __name__ == "__main__":
    # We are simulating n games accumulating a total score
    nr_of_games = int(sys.argv[1])
    print('We will replay {} times in total.'.format(nr_of_games))

    player = SimplePlayer()
    other_players = []

    player_ai = sys.argv[2].lower()
    if player_ai == "mc7":
        player = MonteCarloPlayer7(verbose=True)
        other_players = [NewSimplePlayer(verbose=False) for player_idx in range(3)]
    elif player_ai == "mc":
        player = MonteCarloPlayer(verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    elif player_ai == "rider":
        player = RiderPlayer(policy=policy_value_fn, c_puct=1400, verbose=True)
        other_players = [NewSimplePlayer(verbose=False) for _ in range(3)]
    elif player_ai == "new_simple":
        player = NewSimplePlayer(verbose=True)
        other_players = [SimplePlayer(verbose=False) for _ in range(3)]
    else:
        player = SimplePlayer(verbose=False)
        other_players = [StupidPlayer(verbose=False) for _ in range(3)]

    card_set = sys.argv[3].lower()

    setting_cards = None
    if card_set == "small":
        sub_card_set = sys.argv[4].lower()

        setting_cards = read_card_games("game/game_0008/{}/game_*.pkl".format(sub_card_set))
    elif card_set == "big":
        sub_card_set = sys.argv[4].lower()

        setting_cards = read_card_games("game/game_0032/{}/game_*.pkl".format(sub_card_set))
    else:
        setting_cards = get_setting_cards()

    evaluate_players(nr_of_games, other_players+[player], setting_cards, is_rotating=False)
