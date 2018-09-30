import sys
import time

import copy

from card import Deck
from card import card_to_bitmask

from simulated_player import MonteCarloPlayer5

from simple_game import run_one_step
from strategy_play import greedy_choose
from expert_play import expert_choose


class NewSimplePlayer(MonteCarloPlayer5):
    def play_card(self, game, other_info={}, simulation_time_limit=1):
        for player_idx, suits in other_info.get("lacking_info", {}).items():
            for suit in suits:
                game.lacking_cards[player_idx][suit] = True

            self.say("Player-{} may lack of {} suit({}, {})", player_idx, suit, game.lacking_cards[player_idx], other_info)

        self.say("Player-{}, the information of lacking_card is {}", \
            self.position, [(player_idx, k) for player_idx, info in enumerate(game.lacking_cards) for k, v in info.items() if v])

        hand_cards = [[] if player_idx != self.position else game._player_hands[player_idx] for player_idx in range(4)]

        remaining_cards = Deck().cards
        for card in self.seen_cards + hand_cards[self.position]:
            remaining_cards.remove(card)

        taken_cards = []
        for player_idx, cards in enumerate(game._cards_taken):
            taken_cards.append(card_to_bitmask(cards))

        init_trick = [[None, game.trick]]

        void_info = {}
        for player_idx, info in enumerate(game.lacking_cards):
            if player_idx != self.position:
                void_info[player_idx] = info

        must_have = self.transfer_cards

        selection_func = expert_choose
        self.say("proactive_mode: {}, selection_func={}, num_of_cpu={}", self.proactive_mode, selection_func, self.num_of_cpu)

        played_card = run_one_step(game.trick_nr+1, 
                                   self.position, 
                                   copy.deepcopy(init_trick), 
                                   hand_cards, 
                                   game.is_heart_broken, 
                                   game.expose_heart_ace, 
                                   remaining_cards, 
                                   taken_cards, 
                                   None, 
                                   selection_func, 
                                   must_have, 
                                   void_info)

        self.say("pick {} card", played_card)

        return played_card
