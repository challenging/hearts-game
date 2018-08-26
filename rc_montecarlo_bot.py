#coding=UTF-8
import sys

import json
import logging
import numpy as np

from abc import abstractmethod
from pprint import pprint

from websocket import create_connection

from sample_bot import Log, PokerSocket, PokerBot

from game import Game
from player import SimplePlayer
from simulated_player import MonteCarloPlayer, MonteCarloPlayer2

from card import Card, Suit, Rank, Deck
from rules import transform


IS_DEBUG = False
system_log = Log(IS_DEBUG)


class SimpleBot(PokerBot):

    def __init__(self,name):
        super(SimpleBot, self).__init__(name)

        self.player = MonteCarloPlayer2(verbose=True)
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
        print(message)

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
                print(message)

                system_log.show_message(message)
                system_log.save_logs(message)

                break


    def turn_end(self, data):
        super(SimpleBot, self).turn_end(data)

        def find_leading_suit():
            leading_idx = int(len(self.round_cards_history)/4)*4
            leading_card = self.round_cards_history[leading_idx][1]
            current_card = self.round_cards_history[-1][1]

            return leading_idx, leading_card.suit, leading_card.suit != current_card.suit

        leading_idx, is_lacking = None, False
        if self.round_cards_history:
            if len(self.round_cards_history)%4 > 0:
                leading_idx, leading_suit, is_lacking = find_leading_suit()
                if is_lacking:
                    for player_idx, player in enumerate(self.game.players):
                        if player.name == self.round_cards_history[-1][0]:
                            self.game.lacking_cards[player_idx][leading_suit] = True

                            break

            self.game.trick.append(self.round_cards_history[-1][1])
            for player in self.game.players:
                player.seen_cards.append(self.round_cards_history[-1][1])
        else:
            print("found empty self.round_cards_history", self.round_cards_history)

        if is_lacking:
            print("---------  information about 'turn_end' -----------")
            print("trick, leading_suit, played_suit = {}, {}".format(self.game.trick, self.round_cards_history[leading_idx], self.round_cards_history[-1]))
            for player_idx, lacking_info in enumerate(self.game.lacking_cards):
                print(player_idx, self.game.players[player_idx].name, lacking_info)
            print("---------------------------------------------------")

        #print(self.round_cards_history)
        #print(self.pick_history)


    def pick_card(self, data):
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

        deck = Deck()
        remaining_cards = []
        for c in deck.cards:
            for pc in self.my_hand_cards + self.game.players[self.player.position].seen_cards:
                if c == pc:
                    break
            else:
                remaining_cards.append(c)

        #print("status", len(self.my_hand_cards), len(self.game.players[self.player.position].seen_cards), self.player.position)
        #print("remaining_cards", remaining_cards, len(remaining_cards))
        if remaining_cards:
            if len(self.game.trick) > 0:
                for idx in range(1, len(self.game.trick)+1):
                    player_idx = self.player.position-idx
                    if player_idx >= 0:
                        self.game._player_hands[player_idx] = np.random.choice(remaining_cards, len(self.my_hand_cards)-1, replace=False).tolist()
                    else:
                        player_idx = 4-abs(player_idx)
                        self.game._player_hands[player_idx] = np.random.choice(remaining_cards, len(self.my_hand_cards)-1, replace=False).tolist()

                    for used_card in self.game._player_hands[player_idx]:
                        remaining_cards.remove(used_card)

                    #print(self.player.position, idx, player_idx, ">>>>>", self.game._player_hands[player_idx], len(self.game._player_hands[player_idx]))

        #print("remaining_cards", remaining_cards, len(remaining_cards))
        if remaining_cards:
            if len(self.game.trick) < 3:
                for idx in range(1, 4-len(self.game.trick)):
                    player_idx = self.player.position+idx
                    if player_idx <= 3:
                        self.game._player_hands[player_idx] = np.random.choice(remaining_cards, len(self.my_hand_cards), replace=False).tolist()
                    else:
                        player_idx = player_idx-4
                        self.game._player_hands[player_idx] = np.random.choice(remaining_cards, len(self.my_hand_cards), replace=False).tolist()

                    for used_card in self.game._player_hands[player_idx]:
                        remaining_cards.remove(used_card)

                    #print(self.player.position, idx, player_idx, "<<<<<", self.game._player_hands[player_idx])

        self.game.current_player_idx = self.player.position
        self.game._player_hands[self.player.position] = self.my_hand_cards
        played_card = self.game.players[self.player.position].play_card(self.game._player_hands[self.player.position], self.game)

        message = "Pick Card:{} ({})".format(played_card, candidate_cards)
        print(message)

        system_log.show_message(message)
        system_log.save_logs(message)

        return str(played_card)


    def expose_my_cards(self, data):
        self.my_hand_cards = []
        for card in data["self"]["cards"]:
            self.my_hand_cards.append(transform(card[0], card[1]))

        expose_card = []
        for card in self.my_hand_cards:
            if card == Card(Suit.hearts, Rank.ace):
                expose_card.append(str(card))

                break

        message = "Expose Cards:{}".format(expose_card)
        system_log.show_message(message)
        system_log.save_logs(message)

        return expose_card


    def expose_cards_end(self,data):
        expose_player = None
        expose_card = None

        self.player_names = []
        players = []

        for player in data['players']:
            try:
                if player['exposedCards'] != [] and len(player['exposedCards']) > 0 and player['exposedCards'] is not None:
                    expose_player = player['playerName']
                    expose_card = player['exposedCards']

                self.player_names.append(player["playerName"])

                p = None
                if self.player_names[-1] == self.player_name:
                    p = MonteCarloPlayer2(verbose=True)
                    self.player = p
                else:
                    p = SimplePlayer(verbose=False)

                p.name = self.player_names[-1]
                players.append(p)
            except Exception as e:
                system_log.show_message(e)
                system_log.save_logs(e)

        self.game = Game(players, verbose=False)

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


    def new_round(self, data):
        try:
            if self.game.trick_nr == 0:
                for player_idx in range(4):
                    self.game.players[player_idx].seen_cards = []
        except Exception as e:
            system_log.show_message(e)
            system_log.save_logs(e)


    def round_end(self, data):
        try:
            for player_idx, player_name in enumerate(self.player_names):
                if player_name == data["roundPlayer"]:
                    break

            print("---------  information about 'round_end' ----------")
            print("          status of game: trick_nr={}, trick={}".format(self.game.trick_nr, self.game.trick))
            print("the winner of this trick: {}({})".format(player_name, player_idx))
            print("---------------------------------------------------")

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
            system_log.save_logs(e)


    def deal_end(self,data):
        self.game.score()
        print("---------  current status of game -----------")
        for player_idx, (taken_cards, score) in enumerate(zip(self.game._cards_taken, self.game.player_scores)):
            print(player_idx, taken_cards, score)
        print("---------------------------------------------")

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
        token= sys.argv[3]
        connect_url = sys.argv[4]
    else:
        player_name = "RungChiChen-MonteCarlo"
        player_number = 3
        token = "12345678"
        connect_url = "ws://localhost:8080/"

    sample_bot = SimpleBot(player_name)
    myPokerSocket = PokerSocket(player_name, player_number, token, connect_url, sample_bot)
    myPokerSocket.doListen()

if __name__ == "__main__":
    main()
