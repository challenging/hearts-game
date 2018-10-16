import sys

from sample_bot import PokerSocket
from brain_bot import BrainBot

from new_simulated_player import MonteCarloPlayer7
from mcts_player import MCTSPlayer

def main():
    argv_count=len(sys.argv)

    if argv_count > 2:
        player_name = sys.argv[1]
        player_number = sys.argv[2]

        token = sys.argv[3]
        connect_url = sys.argv[4]
    else:
        player_name = "RungChiChen"
        player_number = 4

        token = "12345678"
        connect_url = "ws://localhost:8080/"

    bot = BrainBot(player_name, MCTSPlayer(verbose=True))
    print("use the player - {}".format(type(bot.player).__name__))

    myPokerSocket = PokerSocket(player_name, player_number, token, connect_url, bot)
    myPokerSocket.doListen()


if __name__ == "__main__":
    main()
