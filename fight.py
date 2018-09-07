import sys

from sample_bot import PokerSocket
from brain_bot import BrainBot

from player import SimplePlayer
from simulated_player import MonteCarloPlayer4, MonteCarloPlayer5
from mcts_player import MCTSPlayer

def main():
    argv_count=len(sys.argv)

    if argv_count>2:
        player_name = sys.argv[1]
        player_number = sys.argv[2]

        token = sys.argv[3]
        connect_url = sys.argv[4]
    else:
        player_name = "RungChiChen"
        player_number = 4

        token = "12345678"
        connect_url = "ws://localhost:8080/"

    bot = BrainBot(player_name, MonteCarloPlayer4(verbose=True))
    if len(sys.argv) == 6:
        if sys.argv[5] == "mc4":
            bot = BrainBot(player_name, MonteCarloPlayer4(verbose=True))
        elif sys.argv[5] == "mcts":
            bot = BrainBot(player_name, MCTSPlayer(verbose=True))
        elif sys.argv[5] == "simple":
            bot = BrainBot(player_name, SimplePlayer(verbose=True))

    print("use the player - {}".format(type(bot.player).__name__))

    myPokerSocket = PokerSocket(player_name, player_number, token, connect_url, bot)
    myPokerSocket.doListen()


if __name__ == "__main__":
    main()
