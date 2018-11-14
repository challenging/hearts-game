import sys

from sample_bot import PokerSocket
from brain_bot import BrainBot

from mcts import policy_value_fn
from new_simulated_player import MonteCarloPlayer7
from dragon_rider_player import RiderPlayer

def main():
    argv_count=len(sys.argv)

    if argv_count > 2:
        player_name = sys.argv[1]
        player_number = sys.argv[2]

        token = sys.argv[3]
        connect_url = sys.argv[4]
    else:
        player_name = "rider bot - 3"
        player_number = 3

        token = "1234567"
        connect_url = "ws://localhost:8080/"

    bot = BrainBot(player_name, RiderPlayer(policy=policy_value_fn, c_puct=1400, verbose=True))
    #bot = BrainBot(player_name, MonteCarloPlayer7(verbose=True))
    print("use the player - {}".format(type(bot.player).__name__))

    myPokerSocket = PokerSocket(player_name, player_number, token, connect_url, bot)
    myPokerSocket.doListen()


if __name__ == "__main__":
    main()
