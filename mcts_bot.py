#coding=UTF-8
import sys

import json
import logging
import numpy as np

from scipy.stats import describe
from abc import abstractmethod
from pprint import pprint

from websocket import create_connection

from sample_bot import Log, PokerSocket, PokerBot

from game import Game
from player import SimplePlayer
from mcts_player import MCTSPlayer

from card import Card, Suit, Rank, Deck
from rules import transform

from montecarlo_bot import MonteCarloBot


IS_DEBUG = False
system_log = Log(IS_DEBUG)


class MCTSBot(MonteCarloBot):
    def __init__(self, name):
        super(MCTSBot, self).__init__(name)

        self.player = MCTSPlayer(verbose=True)


    def expose_cards_end(self, data):
        expose_player, expose_card = None, None

        self.player_names, players = [], []
        for player in data['players']:
            try:
                if player['exposedCards'] != [] and len(player['exposedCards']) > 0 and player['exposedCards'] is not None:
                    expose_player = player['playerName']
                    expose_card = player['exposedCards']

                self.player_names.append(player["playerName"])

                p = None
                if self.player_names[-1] == self.player_name:
                    p = MCTSPlayer(verbose=True)
                    self.player = p
                else:
                    p = SimplePlayer(verbose=False)

                p.name = self.player_names[-1]
                players.append(p)
            except Exception as e:
                system_log.show_message(e)
                system_log.save_logs(e)

        self.game = Game(players, verbose=False)

        idx, deal_number = None, data["dealNumber"]
        if deal_number == 1:
            idx = (self.player.position+1)%4
            self.player.set_transfer_card(idx, self.given_cards)
        elif deal_number == 2:
            idx = (self.player.position+3)%4
            self.player.set_transfer_card(idx, self.given_cards)
        elif deal_number == 3:
            idx = (self.player.position+2)%4
            self.player.set_transfer_card(idx, self.given_cards)

        if idx is not None:
            self.


        if expose_player is not None and expose_card is not None:
            message="Player:{}, Expose card:{}".format(expose_player, expose_card)
            system_log.show_message(message)
            system_log.save_logs(message)
            self.expose_card = True
        else:
            message="No player expose card!"
            system_log.show_message(message)
            system_log.save_logs(message)
            self.expose_card=False


def main():
    argv_count=len(sys.argv)

    if argv_count>2:
        player_name = sys.argv[1]
        player_number = sys.argv[2]

        token = sys.argv[3]
        connect_url = sys.argv[4]
    else:
        player_name = "RungChiChen-MCTS"
        player_number = 4

        token = "12345678"
        connect_url = "ws://localhost:8080/"

    bot = MCTSBot(player_name)
    myPokerSocket = PokerSocket(player_name, player_number, token, connect_url, bot)
    myPokerSocket.doListen()

if __name__ == "__main__":
    main()
