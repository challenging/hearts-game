#coding=UTF-8
from abc import abstractmethod

from websocket import create_connection
import json
import logging
import sys

from sample_bot import Log, PokerSocket, PokerBot

from game import Game
from player import SimplePlayer
from simulated_player import MonteCarloPlayer

from card import Card, Suit, Rank
from rules import transform


IS_DEBUG = False
system_log = Log(IS_DEBUG)


class SimpleBot(PokerBot):

    def __init__(self,name):
        super(SimpleBot, self).__init__(name)

        self.player = SimplePlayer(verbose=False)
        self.player_names = []

        self.game = None

        self.my_hand_cards = []
        self.expose_card = False
        self.my_pass_card = []


    def receive_cards(self, data):
        self.my_hand_cards = self.get_cards(data)


    def pass_cards(self, data):
        cards = data['self']['cards']
        self.my_hand_cards = []

        for card_str in cards:
            card = transform(card_str[0], card_str[1])
            self.my_hand_cards.append(card)

        pass_cards = self.player.pass_cards(self.my_hand_cards)

        return_values = []
        for card in pass_cards:
            return_values.append(str(card))

        message="Pass Cards:{}".format(return_values)

        system_log.show_message(message)
        system_log.save_logs(message)

        self.my_pass_card = return_values

        return return_values


    def receive_opponent_cards(self,data):
        self.my_hand_cards = self.get_cards(data)

        players = data['players']
        for player in players:
            player_name = player['playerName']
            if player_name == self.player_name:
                picked_cards = player['pickedCards']
                receive_cards = player['receivedCards']

                message = "User Name:{}, Picked Cards:{}, Receive Cards:{}".format(player_name, picked_cards, receive_cards)

                system_log.show_message(message)
                system_log.save_logs(message)

                break


    def turn_end(self, data):
        super(SimpleBot, self).turn_end(data)

        self.game.trick.append(self.round_cards_history[-1][1])


    def pick_card(self,data):
        candidate_cards = []
        for card in data['self']['candidateCards']:
            candidate_cards.append(transform(card[0], card[1]))

        self.my_hand_cards = []
        for card_str in data['self']['cards']:
            card = transform(card_str[0], card_str[1])
            self.my_hand_cards.append(card)

        message = "My Cards:{}".format(self.my_hand_cards)
        system_log.show_message(message)
        message = "Pick Card Event Content:{}".format(data)
        system_log.show_message(message)
        message = "Candidate Cards:{}".format(candidate_cards)
        system_log.show_message(message)
        system_log.save_logs(message)

        played_card = self.player.play_card(candidate_cards, self.game)
        message = "Pick Card:{} ({})".format(played_card, candidate_cards)

        system_log.show_message(message)
        system_log.save_logs(message)

        return str(played_card)


    def expose_my_cards(self, data):
        self.my_hand_cards = []
        for card in data["self"]["cards"]:
            self.my_hand_cards.append(transform(card[0], card[1]))

        expose_card = []
        for card in self.my_hand_cards:
            if card == Card(Suit.spades, Rank.queen):
                expose_card.append(str(card))

        message = "Expose Cards:{}".format(expose_card)
        system_log.show_message(message)
        system_log.save_logs(message)

        return expose_card


    def expose_cards_end(self,data):
        players = data['players']
        expose_player = None
        expose_card = None

        self.player_names = []
        self.game = Game([SimplePlayer(verbose=False) for idx in range(4)], verbose=False)

        for player in players:
            try:
                if player['exposedCards'] != [] and len(player['exposedCards']) > 0 and player['exposedCards'] is not None:
                    expose_player = player['playerName']
                    expose_card = player['exposedCards']

                self.player_names.append(player["playerName"])
            except Exception as e:
                system_log.show_message(e)
                system_log.save_logs(e)

        if expose_player is not None and expose_card is not None:
            message="Player:{}, Expose card:{}".format(expose_player,expose_card)
            system_log.show_message(message)
            system_log.save_logs(message)
            self.expose_card = True
        else:
            message="No player expose card!"
            system_log.show_message(message)
            system_log.save_logs(message)
            self.expose_card=False


    def new_round(self, data):
        try:
            first_player_name = data["roundPlayers"][0]
            first_player_idx, self_player_idx = None, None
            for player_idx, player_name in enumerate(self.player_names):
                if player_name == first_player_name:
                    first_player_idx = player_idx

                if player_name == self.player_name:
                    self_player_idx = player_idx

            self.game.current_player_idx = first_player_idx

            hand_cards = []
            for card_str in data['self']['cards']:
                hand_cards.append(transform(card_str[0], card_str[1]))

            self.game._player_hands = [hand_cards if idx == self_player_idx else [] for idx in range(4)]
        except Exception as e:
            system_log.show_message(e)


    def round_end(self, data):
        try:
            for player_idx, player_name in enumerate(self.player_names):
                if player_name == data["roundPlayer"]:
                    break

            self.game._cards_taken[player_idx].extend([card for _, card in self.round_cards_history[self.game.trick_nr*4:(self.game.trick_nr+1)*4]])

            self.game.trick_nr += 1
            self.game.trick = []

            round_scores = self.get_round_scores(self.expose_card, data)
            for key in round_scores.keys():
                message = "Player name:{}, Round score:{}".format(key, round_scores.get(key))
                system_log.show_message(message)
                system_log.save_logs(message)
        except Exception as e:
            system_log.show_message(e)


    def deal_end(self,data):
        self.game.reset()

        self.my_hand_cards = []
        self.expose_card = False
        deal_scores,initial_cards,receive_cards,picked_cards=self.get_deal_scores(data)
        message = "Player name:{}, Pass Cards:{}".format(self.player_name, self.my_pass_card)
        system_log.show_message(message)
        system_log.save_logs(message)

        for key in deal_scores.keys():
            message = "Player name:{}, Deal score:{}".format(key,deal_scores.get(key))
            system_log.show_message(message)
            system_log.save_logs(message)

        for key in initial_cards.keys():
            message = "Player name:{}, Initial cards:{}, Receive cards:{}, Picked cards:{}".format(key, initial_cards.get(key),receive_cards.get(key),picked_cards.get(key))
            system_log.show_message(message)
            system_log.save_logs(message)


    def game_over(self,data):
        game_scores = self.get_game_scores(data)
        for key in game_scores.keys():
            message = "Player name:{}, Game score:{}".format(key, game_scores.get(key))
            system_log.show_message(message)
            system_log.save_logs(message)


    def pick_history(self,data,is_timeout,pick_his):
        for key in pick_his.keys():
            message = "Player name:{}, Pick card:{}, Is timeout:{}".format(key,pick_his.get(key),is_timeout)
            system_log.show_message(message)
            system_log.save_logs(message)


def main():
    argv_count=len(sys.argv)

    if argv_count>2:
        player_name = sys.argv[1]
        player_number = sys.argv[2]

        token = "12345678"
        connect_url = "ws://localhost:8080/"
    else:
        player_name = "RungChiChen-Simple"
        player_number = 1

        token = "12345678"
        connect_url = "ws://localhost:8080/"

    sample_bot = SimpleBot(player_name)
    myPokerSocket = PokerSocket(player_name, player_number, token, connect_url, sample_bot)
    myPokerSocket.doListen()

if __name__ == "__main__":
    main()
