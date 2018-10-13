#coding=UTF-8
import os
import sys

import json
import logging

from scipy.stats import describe

from abc import abstractmethod
from websocket import create_connection

from card import Card, Suit, Rank
from card import transform


class Log(object):
    def __init__(self,is_debug=True):
        self.is_debug=is_debug
        self.msg=None
        self.logger = logging.getLogger('hearts_logs')

        if os.path.exists("/log"):
            hdlr = logging.FileHandler('/log/hearts_logs.log')
        else:
            hdlr = logging.FileHandler('hearts_logs.log')

        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def show_message(self,msg):
        if self.is_debug:
            print(msg)

    def save_logs(self,msg):
        self.logger.info(msg)

IS_DEBUG = False
system_log = Log(IS_DEBUG)

OUT_FILE = None


class PokerBot(object):

    def __init__(self,player_name):
        self.round_cards_history = []
        self.pick_his = {}
        self.round_cards = {}
        self.score_cards = {}
        self.player_name = player_name
        self.players_current_picked_cards = []

        self.verbose = True

        self.game_score_cards = {Card(Suit.spades, Rank.queen), Card(Suit.clubs, Rank.ten)}
        for idx in range(2, 15):
            rank = None

            if idx == 2:
                rank = Rank.two
            elif idx == 3:
                rank = Rank.three
            elif idx == 4:
                rank = Rank.four
            elif idx == 5:
                rank = Rank.five
            elif idx == 6:
                rank = Rank.six
            elif idx == 7:
                rank = Rank.seven
            elif idx == 8:
                rank = Rank.eight
            elif idx == 9:
                rank = Rank.nine
            elif idx == 10:
                rank = Rank.ten
            elif idx == 11:
                rank = Rank.jack
            elif idx == 12:
                rank = Rank.queen
            elif idx == 13:
                rank = Rank.king
            elif idx == 14:
                rank = Rank.ace

            self.game_score_cards.add(Card(Suit.hearts, rank))


    def say(self, message, *formatargs):
        if self.verbose:
            global OUT_FILE

            message = message.format(*formatargs)
            if not os.path.exists("/log"):
                print(message)
            else:
                if OUT_FILE is None:
                    OUT_FILE = open("/log/bot.log", "a")

                OUT_FILE.write("{}\n".format(message))


    def receive_cards(self,data):
        err_msg = self.__build_err_msg("receive_cards")
        raise NotImplementedError(err_msg)


    def pass_cards(self, data):
        err_msg = self.__build_err_msg("pass_cards")
        raise NotImplementedError(err_msg)


    def pick_card(self,data):
        err_msg = self.__build_err_msg("pick_card")
        raise NotImplementedError(err_msg)

    def expose_my_cards(self,yourcards):
        err_msg = self.__build_err_msg("expose_my_cards")
        raise NotImplementedError(err_msg)


    def expose_cards_end(self,data):
        err_msg = self.__build_err_msg("expose_cards_announcement")
        raise NotImplementedError(err_msg)


    def receive_opponent_cards(self,data):
        err_msg = self.__build_err_msg("receive_opponent_cards")
        raise NotImplementedError(err_msg)

    def new_round(self, data):
        err_msg = self.__build_err_msg("new_round")
        raise NotImplementedError(err_msg)

    def round_end(self,data):
        err_msg = self.__build_err_msg("round_end")
        raise NotImplementedError(err_msg)


    def deal_end(self,data):
        err_msg = self.__build_err_msg("deal_end")
        raise NotImplementedError(err_msg)


    def game_over(self,data):
        err_msg = self.__build_err_msg("game_over")
        raise NotImplementedError(err_msg)


    def pick_history(self,data,is_timeout,pick_his):
        err_msg = self.__build_err_msg("pick_history")
        raise NotImplementedError(err_msg)


    def reset_card_his(self):
        self.round_cards_history = []
        self.pick_his={}


    def get_card_history(self):
        return self.round_cards_history


    def turn_end(self,data):
        turnCard = data['turnCard']
        turnCard = transform(turnCard[0], turnCard[1])

        turnPlayer = data['turnPlayer']
        players = data['players']
        is_timeout = data['serverRandom']

        for player in players:
            player_name = player['playerName']
            if player_name == self.player_name:
                current_cards = player['cards']
                for card in current_cards:
                    self.players_current_picked_cards.append(transform(card[0], card[1]))

                break

        self.round_cards[turnPlayer] = turnCard

        opp_pick = {}
        opp_pick[turnPlayer] = turnCard

        if (self.pick_his.get(turnPlayer)) is not None:
            pick_card_list=self.pick_his.get(turnPlayer)
            pick_card_list.append(turnCard)
            self.pick_his[turnPlayer]=pick_card_list
        else:
            pick_card_list = []
            pick_card_list.append(turnCard)
            self.pick_his[turnPlayer] = pick_card_list

        self.round_cards_history.append((turnPlayer, turnCard))
        self.pick_history(data, is_timeout, opp_pick)

    def get_cards(self, data):
        try:
            receive_cards = []
            players = data['players']
            for player in players:
                if player['playerName'] == self.player_name:
                    cards = player['cards']
                    for card in cards:
                        receive_cards.append(transform(card[0], card[1]))
                    break
            return receive_cards
        except Exception as e:
            system_log.show_message(e)

            return None

    def get_round_scores(self,is_expose_card=False,data=None):
        if data!=None:
            players=data['roundPlayers']
            picked_user = players[0]
            round_card = self.round_cards.get(picked_user)
            score_cards = []
            for i in range(len(players)):
                card = self.round_cards.get(players[i])
                if card in self.game_score_cards:
                    score_cards.append(card)

                if round_card.suit == card.suit:
                    if round_card.rank < card.rank:
                        picked_user = players[i]
                        round_card=card

            if (self.score_cards.get(picked_user) is not None):
                current_score_cards=self.score_cards.get(picked_user)
                score_cards+=current_score_cards

            self.score_cards[picked_user]=score_cards
            self.round_cards = {}

        receive_cards={}
        for key in self.pick_his.keys():
            picked_score_cards=self.score_cards.get(key)
            round_score = 0
            round_heart_score=0
            is_double = False
            if picked_score_cards!=None:
                for card in picked_score_cards:
                    if card in self.game_score_cards:
                        if card == Card(Suit.spades, Rank.queen):
                            round_score += -13
                        elif card == Card(Suit.clubs, Rank.ten):
                            is_double = True
                        else:
                            round_heart_score += -1
                if is_expose_card:
                    round_heart_score*=2
                round_score+=round_heart_score
                if is_double:
                    round_score*=2
            receive_cards[key] = round_score
        return receive_cards

    def get_deal_scores(self, data):
        try:
            self.score_cards = {}
            final_scores  = {}
            initial_cards = {}
            receive_cards = {}
            picked_cards  = {}
            players = data['players']
            for player in players:
                player_name     = player['playerName']
                palyer_score    = player['dealScore']
                player_initial  = player['initialCards']
                player_receive  = player['receivedCards']
                player_picked   = player['pickedCards']

                final_scores[player_name] = palyer_score
                initial_cards[player_name] = player_initial
                receive_cards[player_name]=player_receive
                picked_cards[player_name]=player_picked
            return final_scores, initial_cards,receive_cards,picked_cards
        except Exception as e:
            system_log.show_message(e)
            return None

    def get_game_scores(self,data):
        try:
            receive_cards={}
            players=data['players']
            for player in players:
                player_name=player['playerName']
                palyer_score=player['gameScore']
                receive_cards[player_name]=palyer_score
            return receive_cards
        except Exception as e:
            system_log.show_message(e)
            return None

class PokerSocket(object):
    ws = ""
    def __init__(self,player_name,player_number,token,connect_url,poker_bot):
        self.player_name=player_name
        self.connect_url=connect_url
        self.player_number=player_number
        self.poker_bot=poker_bot
        self.token=token

    def takeAction(self,action, data):
       #print(action)

       if action=="new_deal":
           self.poker_bot.receive_cards(data)
       elif action=="pass_cards":
           pass_cards=self.poker_bot.pass_cards(data)
           self.ws.send(json.dumps(
                {
                    "eventName": "pass_my_cards",
                    "data": {
                        "dealNumber": data['dealNumber'],
                        "cards": pass_cards
                    }
                }))
       elif action=="receive_opponent_cards":
           self.poker_bot.receive_opponent_cards(data)
       elif action=="expose_cards":
           export_cards = self.poker_bot.expose_my_cards(data)
           if export_cards!=None:
               self.ws.send(json.dumps(
                   {
                       "eventName": "expose_my_cards",
                       "data": {
                           "dealNumber": data['dealNumber'],
                           "cards": export_cards
                       }
                   }))
       elif action == "expose_cards_end":
           self.poker_bot.expose_cards_end(data)
       elif action == "new_round":
           self.poker_bot.new_round(data)
       elif action=="your_turn":
           pick_card = self.poker_bot.pick_card(data)
           message="Send message:{}".format(json.dumps(
                {
                   "eventName": "pick_card",
                   "data": {
                       "dealNumber": data['dealNumber'],
                       "roundNumber": data['roundNumber'],
                       "turnCard": pick_card
                   }
               }))

           system_log.show_message(message)
           system_log.save_logs(message)
           self.ws.send(json.dumps(
               {
                   "eventName": "pick_card",
                   "data": {
                       "dealNumber": data['dealNumber'],
                       "roundNumber": data['roundNumber'],
                       "turnCard": pick_card
                   }
               }))
       elif action=="turn_end":
           self.poker_bot.turn_end(data)
       elif action=="round_end":
           self.poker_bot.round_end(data)
       elif action=="deal_end":
           self.poker_bot.deal_end(data)
           self.poker_bot.reset_card_his()
       elif action=="game_end":
           self.poker_bot.game_over(data)
           self.ws.send(json.dumps({
               "eventName": "stop_game",
               "data": {}
           }))
           self.ws.close()

    def doListen(self):
        try:
            self.ws = create_connection(self.connect_url)
            self.ws.send(json.dumps({
                "eventName": "join",
                "data": {
                    "playerNumber":self.player_number,
                    "playerName":self.player_name,
                    "token":self.token
                }
            }))
            while 1:
                result = self.ws.recv()
                msg = json.loads(result)
                event_name = msg["eventName"]
                data = msg["data"]
                system_log.show_message(event_name)
                system_log.save_logs(event_name)
                system_log.show_message(data)
                system_log.save_logs(data)
                self.takeAction(event_name, data)
        except Exception as e:
            system_log.show_message(e)
            self.doListen()

class LowPlayBot(PokerBot):

    def __init__(self,name):
        super(LowPlayBot,self).__init__(name)
        self.my_hand_cards=[]
        self.expose_card=False
        self.my_pass_card=[]

    def receive_cards(self,data):
        self.my_hand_cards=self.get_cards(data)

    def pass_cards(self,data):
        cards = data['self']['cards']
        self.my_hand_cards = []
        for card_str in cards:
            card = Card(card_str[1], card_str[0])
            self.my_hand_cards.append(card)
        pass_cards=[]
        count=0
        for i in range(len(self.my_hand_cards)):
            card=self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
            if card == Card(Suit.spades, Rank.queen):
                pass_cards.append(card)
                count+=1
            elif card == Card(Suit.clubs, Rank.ten):
                pass_cards.append(card)
                count += 1
        for i in range(len(self.my_hand_cards)):
            card = self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
            if card.suit_index==2:
                pass_cards.append(card)
                count += 1
                if count ==3:
                    break
        if count <3:
            for i in range(len(self.my_hand_cards)):
                card = self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
                if card not in self.game_score_cards:
                    pass_cards.append(card)
                    count += 1
                    if count ==3:
                        break
        return_values=[]
        for card in pass_cards:
            return_values.append(card.toString())
        message="Pass Cards:{}".format(return_values)
        system_log.show_message(message)
        system_log.save_logs(message)
        self.my_pass_card=return_values
        return return_values

    def pick_card(self,data):
        cadidate_cards=data['self']['candidateCards']
        cards = data['self']['cards']
        self.my_hand_cards = []
        for card_str in cards:
            card = Card(card_str[1], card_str[0])
            self.my_hand_cards.append(card)
        message = "My Cards:{}".format(self.my_hand_cards)
        system_log.show_message(message)
        card_index=0
        message = "Pick Card Event Content:{}".format(data)
        system_log.show_message(message)
        message = "Candidate Cards:{}".format(cadidate_cards)
        system_log.show_message(message)
        system_log.save_logs(message)
        message = "Pick Card:{}".format(cadidate_cards[card_index])
        system_log.show_message(message)
        system_log.save_logs(message)
        return cadidate_cards[card_index]

    def expose_my_cards(self,yourcards):
        expose_card=[]
        for card in self.my_hand_cards:
            if card == Card(Suit.hearts, Rank.ace):
                expose_card.append(card.toString())

        message = "Expose Cards:{}".format(expose_card)
        system_log.show_message(message)
        system_log.save_logs(message)
        return expose_card

    def expose_cards_end(self,data):
        players = data['players']
        expose_player=None
        expose_card=None
        for player in players:
            try:
                if player['exposedCards']!=[] and len(player['exposedCards'])>0 and player['exposedCards']!=None:
                    expose_player=player['playerName']
                    expose_card=player['exposedCards']
            except Exception as e:
                system_log.show_message(e)
                system_log.save_logs(e)
        if expose_player!=None and expose_card!=None:
            message="Player:{}, Expose card:{}".format(expose_player,expose_card)
            system_log.show_message(message)
            system_log.save_logs(message)
            self.expose_card=True
        else:
            message="No player expose card!"
            system_log.show_message(message)
            system_log.save_logs(message)
            self.expose_card=False

    def receive_opponent_cards(self,data):
        self.my_hand_cards = self.get_cards(data)
        players = data['players']
        for player in players:
            player_name = player['playerName']
            if player_name == self.player_name:
                picked_cards = player['pickedCards']
                receive_cards = player['receivedCards']
                message = "User Name:{}, Picked Cards:{}, Receive Cards:{}".format(player_name, picked_cards,receive_cards)
                system_log.show_message(message)
                system_log.save_logs(message)

    def round_end(self,data):
        try:
            round_scores=self.get_round_scores(self.expose_card, data)
            for key in round_scores.keys():
                message = "Player name:{}, Round score:{}".format(key, round_scores.get(key))
                system_log.show_message(message)
                system_log.save_logs(message)
        except Exception as e:
            system_log.show_message(e)

    def deal_end(self,data):
        self.my_hand_cards=[]
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
        player_name="Eric"
        player_number=4
        token="12345678"
        connect_url="ws://localhost:8080/"

    sample_bot=LowPlayBot(player_name)
    myPokerSocket=PokerSocket(player_name,player_number,token,connect_url,sample_bot)
    myPokerSocket.doListen()

if __name__ == "__main__":
    main()
