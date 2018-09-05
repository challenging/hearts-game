import sys

from sample_bot import PokerSocket
from simple_bot import SimpleBot
from mcts_bot import MCTSBot
from montecarlo_bot import MonteCarloBot

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

    bot = SimpleBot(player_name)
    if len(sys.argv) == 6:
        if sys.argv[5] == "mcts":
            bot = MCTSBot(player_name)
        elif sys.argv[5] == "mc":
            bot = MonteCarloBot(player_name)
        else:
            print("use the default player - SimpleBot")

    myPokerSocket = PokerSocket(player_name, player_number, token, connect_url, bot)
    myPokerSocket.doListen()

if __name__ == "__main__":
    main()
