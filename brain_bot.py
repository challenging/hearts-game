#coding=UTF-8
import sys

import json
import copy
import time

import logging
import numpy as np

from abc import abstractmethod
from pprint import pprint
from collections import defaultdict, deque
from scipy.stats import describe

from websocket import create_connection

from sample_bot import Log, LowPlayBot, PokerSocket

from game import Game
from player import SimplePlayer

from card import Card, Suit, Rank, Deck
from card import transform


IS_DEBUG = False
ALL_SCORES = defaultdict(list)

system_log = Log(IS_DEBUG)


class BrainBot(LowPlayBot):
    def __init__(self, name, brain):
        super(BrainBot, self).__init__(name)

        self.player = brain
        self.pure_player = copy.deepcopy(brain)
        self.proactive_mode = set()

        self.player_names = []

        self.game = None
        self.prev_turn_end_time = None

        self.other_info = {}
        self.decision_time_info = {}

        self.early_lacking = False
        self.num_early_lacking = 0
        self.num_shooting_moon = 0
        self.num_shooting_moon_lacking_info = 0

        self.given_cards = []
        self.received_cards = []

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

        pass_cards = self.player.pass_cards(self.my_hand_cards, data["dealNumber"])
        self.proactive_mode = self.player.proactive_mode

        return_values = []
        for card in pass_cards:
            return_values.append(str(card))

        message="Pass Cards:{}".format(return_values)
        system_log.show_message(message)
        system_log.save_logs(message)

        self.my_pass_card = return_values

        return return_values


    def receive_opponent_cards(self, data):
        self.my_hand_cards = self.get_cards(data)

        self.given_cards = []
        self.received_cards = []

        players = data['players']
        for player in players:
            player_name = player['playerName']
            if player_name == self.player_name:
                picked_cards = player['pickedCards']
                for card_str in picked_cards:
                    card = transform(card_str[0], card_str[1])
                    self.given_cards.append(card)

                receive_cards = player['receivedCards']
                for card_str in receive_cards:
                    card = transform(card_str[0], card_str[1])
                    self.received_cards.append(card)

                message = "User Name:{}, Given Cards:{}, Receive Cards:{}".format(player_name, self.given_cards, receive_cards)
                self.say(message)

                system_log.show_message(message)
                system_log.save_logs(message)

                break


    def turn_end(self, data):
        super(BrainBot, self).turn_end(data)

        decision_time = sys.maxsize

        stime = time.time()
        if self.prev_turn_end_time is None:
            self.prev_turn_end_time = stime
        else:
            decision_time = stime-self.prev_turn_end_time

            self.prev_turn_end_time = stime

        def find_leading_suit():
            leading_idx = int(len(self.round_cards_history)/4)*4
            leading_card = self.round_cards_history[leading_idx][1]
            current_card = self.round_cards_history[-1][1]

            #self.say("(leading_card, current_card) = ({}, {})", leading_card, current_card)
            return leading_idx, leading_card.suit, leading_card.suit != current_card.suit

        current_player_name, last_card = self.round_cards_history[-1]

        leading_idx, is_lacking = None, False
        if self.round_cards_history:
            if len(self.round_cards_history)%4 > 0:
            #if current_player_name != self.player_name:
                leading_idx, leading_suit, is_lacking = find_leading_suit()
                #self.say("(leading_idx, leading_suit, is_lacking) = ({}, {}, {}), (player_names, current_player_name) = {}, {})", \
                #    leading_idx, leading_suit, is_lacking, [player.name for player in self.game.players], current_player_name)

                if is_lacking:
                    for player_idx, player in enumerate(self.game.players):
                        if player.name == current_player_name:
                            self.game.lacking_cards[player_idx][leading_suit] = True
                            #self.say("set info of lacking_cards is {}", self.game.lacking_cards)

                            break
                else:
                    if (self.game.trick_nr != 0 and self.game.trick) and current_player_name != self.player_name and len(self.decision_time_info.get(current_player_name, [])) > 2:
                        array = np.array(self.decision_time_info[current_player_name])
                        n = np.sum(array > 1)

                        if decision_time < 0.8 and n > 4:
                            self.other_info.setdefault("lacking_info", {})

                            player_idx = [idx for idx in range(4) if self.player_names[idx] == current_player_name][0]
                            self.other_info["lacking_info"].setdefault(player_idx, [])
                            self.other_info["lacking_info"][player_idx].append(leading_suit)

                            self.say("------> Player-{} lacks of {}({}) because of {:.4f} seconds, {}",\
                                current_player_name, leading_suit, last_card, decision_time, n)

                            if not self.early_lacking and self.game.trick_nr < 6:
                                self.early_lacking = True
                                self.num_early_lacking += 1

                            #print(self.early_lacking, self.num_shooting_moon_lacking_info, self.num_shooting_moon, self.num_early_lacking)

            if isinstance(decision_time, float) and current_player_name != self.player_name and len(self.round_cards_history) > 1 and self.game.trick_nr < 6:
                self.decision_time_info.setdefault(current_player_name, deque(maxlen=8))
                self.decision_time_info[current_player_name].append(decision_time)

                self.say("{}'s decision_time_info is {}", current_player_name, self.decision_time_info[current_player_name])

            self.game.trick.append(last_card)
            self.player.seen_cards.append(last_card)
        else:
            self.say("found empty self.round_cards_history", self.round_cards_history)

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
        self.game._player_hands[self.player.position] = self.my_hand_cards

        message = "My Cards:{}".format(self.game._player_hands[self.player.position])
        #self.say(message)
        system_log.show_message(message)

        message = "Pick Card Event Content:{}".format(data)
        system_log.show_message(message)

        message = "Candidate Cards:{}".format(candidate_cards)
        system_log.show_message(message)
        system_log.save_logs(message)

        #print("players's position", self.player.position, self.player.proactive_mode, self.player.seen_cards)
        #print("current trick", self.game.trick)
        #print("current hand cards", self.game._player_hands[self.player.position])

        """
        deck = Deck()
        for idx in range(1, len(self.game.trick)+1):
            player_idx = (self.player.position+(4-idx))%4
            self.game._player_hands[player_idx] = np.random.choice(deck.cards, len(self.my_hand_cards)-1, replace=False).tolist()

        for idx in range(1, 4-len(self.game.trick)):
            player_idx = (self.player.position+idx)%4
            self.game._player_hands[player_idx] = np.random.choice(deck.cards, len(self.my_hand_cards), replace=False).tolist()
        """

        self.game.current_player_idx = self.player.position
        self.game._player_hands[self.player.position] = self.my_hand_cards
        played_card = self.game.players[self.player.position].play_card(self.game, self.other_info)

        message = "Pick Card:{} from {}".format(played_card, candidate_cards)
        self.say(message)

        system_log.show_message(message)
        system_log.save_logs(message)

        return str(played_card)


    def expose_my_cards(self, data):
        self.my_hand_cards = []
        for card in data["self"]["cards"]:
            self.my_hand_cards.append(transform(card[0], card[1]))

        expose_card = []
        """
        if self.proactive_mode:
            for card in self.my_hand_cards:
                if card == Card(Suit.hearts, Rank.ace):
                    expose_card.append(str(card))

                    break
        """

        message = "Expose Cards:{}".format(expose_card)
        system_log.show_message(message)
        system_log.save_logs(message)

        return expose_card


    def expose_cards_end(self, data):
        expose_player, expose_card = None, None

        self.player_names, current_player_idx, players = [], None, []
        for player_idx, player in enumerate(data['players']):
            try:
                #if player['exposedCards'] != [] and len(player['exposedCards']) > 0 and player['exposedCards'] is not None:
                #    expose_player = player['playerName']
                #    expose_card = player['exposedCards'][0]

                self.player_names.append(player["playerName"])

                p = None
                if self.player_names[-1] == self.player_name:
                    p = copy.deepcopy(self.pure_player)
                    current_player_idx = player_idx
                else:
                    p = SimplePlayer(verbose=False)

                p.name = self.player_names[-1]
                players.append(p)
            except Exception as e:
                system_log.show_message(e)
                system_log.save_logs(e)

        self.game = Game(players, verbose=False)
        self.player = self.game.players[current_player_idx]


        if self.proactive_mode:
            self.player.proactive_mode = self.proactive_mode
        else:
            self.player.evaluate_proactive_mode(self.game)

        idx = None
        deal_number = data["dealNumber"]
        if deal_number%4 == 1:
            idx = (self.player.position+1)%4
            self.player.set_transfer_card(idx, self.given_cards)
        elif deal_number%4 == 2:
            idx = (self.player.position+3)%4
            self.player.set_transfer_card(idx, self.given_cards)
        elif deal_number%4 == 3:
            idx = (self.player.position+2)%4
            self.player.set_transfer_card(idx, self.given_cards)

        if idx is not None:
            self.say("pass card to {}, {}, {}", idx, self.player.transfer_cards, self.given_cards)
        else:
            self.say("not passing card")


        if expose_player is not None and expose_card is not None:
            message="Player:{}, Expose card:{}".format(expose_player, expose_card)
            self.game.expose_heart_ace = True
            self.player.set_transfer_card([idx for idx in range(4) if self.player_names[idx] == expose_player][0], transform(expose_card[0], expose_card[1]))

            #print("expose", message, self.player.transfer_cards)

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

            self.game._cards_taken[player_idx].extend([card for _, card in self.round_cards_history[self.game.trick_nr*4:(self.game.trick_nr+1)*4]])

            if self.player.proactive_mode:
                for player_idx, player in enumerate(data["players"]):
                    if player["playerName"] != self.player_name:
                        score = self.game.count_points(self.game._cards_taken[player_idx])

                        if score > 0:
                            self.say("turn off the proactive mode from {}, {}", self.player.proactive_mode, score)
                            self.player.proactive_mode = set()

                            break

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
        global ALL_SCORES

        self.game.score()
        self.reset_status()

        deal_scores,initial_cards,receive_cards,picked_cards=self.get_deal_scores(data)

        message = "Player name:{}, Pass Cards:{}".format(self.player_name, self.my_pass_card)
        system_log.show_message(message)
        system_log.save_logs(message)

        for key in deal_scores.keys():
            message = "Player name:{}, Deal score:{}".format(key, deal_scores.get(key))

            if key == self.player_name and deal_scores.get(key) > 100:
                self.num_shooting_moon += 1

                if self.early_lacking:
                    self.num_shooting_moon_lacking_info += 1

            system_log.show_message(message)
            system_log.save_logs(message)

            ALL_SCORES[key].append(deal_scores.get(key))

        for key in initial_cards.keys():
            message = "Player name:{}, Initial cards:{}, Receive cards:{}, Picked cards:{}".format(key, initial_cards.get(key),receive_cards.get(key),picked_cards.get(key))

            system_log.show_message(message)
            system_log.save_logs(message)

        for key, scores in ALL_SCORES.items():
            stats = describe(scores)
            self.say("Player - {} gets (num_shooting_moon_lacking_info, num_shooting_moon, num_early_lacking = {},{},{}), (n={}, mean={:.2f}, std={:.2f}, minmax={})", \
                key, self.num_shooting_moon_lacking_info, self.num_shooting_moon, self.num_early_lacking, \
                stats.nobs, stats.mean, stats.variance**0.5, stats.minmax)


    def reset_status(self):
        self.proactive_mode = set()

        self.my_hand_cards = []
        self.given_cards = []
        self.received_cards = []

        self.expose_card = False

        self.prev_turn_end_time = None
        self.other_info = {}

        self.early_lacking = False
