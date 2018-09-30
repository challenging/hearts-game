import sys
import time

import multiprocessing as mp

from scipy.stats import describe
from collections import defaultdict

from card import Deck
from card import card_to_bitmask

from simulated_player import TIMEOUT_SECOND, COUNT_CPU
from simulated_player import MonteCarloPlayer5

from simple_game import run_simulation
from strategy_play import random_choose, greedy_choose
from expert_play import expert_choose


TIMEOUT_SECOND = 0.91


class MonteCarloPlayer7(MonteCarloPlayer5):
    def __init__(self, num_of_cpu=COUNT_CPU, verbose=False):
        super(MonteCarloPlayer7, self).__init__(verbose=verbose)


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        stime = time.time()

        for player_idx, suits in other_info.get("lacking_info", {}).items():
            for suit in suits:
                game.lacking_cards[player_idx][suit] = True

            self.say("Player-{} may lack of {} suit({}, {})", player_idx, suit, game.lacking_cards[player_idx], other_info)

        self.say("Player-{}, the information of lacking_card is {}", self.position, [(player_idx, k) for player_idx, info in enumerate(game.lacking_cards) for k, v in info.items() if v])

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

        played_card = None

        selection_func = greedy_choose #if self.proactive_mode else greedy_choose
        self.say("proactive_mode: {}, selection_func={}, num_of_cpu={}", self.proactive_mode, selection_func, self.num_of_cpu)

        pool = mp.Pool(processes=self.num_of_cpu)
        mul_result = [pool.apply_async(run_simulation, args=(seed,
                                                             game.trick_nr+1, 
                                                             self.position, 
                                                             init_trick, 
                                                             hand_cards, 
                                                             game.is_heart_broken, 
                                                             game.expose_heart_ace, 
                                                             remaining_cards, 
                                                             taken_cards, 
                                                             played_card, 
                                                             selection_func, 
                                                             must_have, 
                                                             void_info, 
                                                             TIMEOUT_SECOND-0.02)) for seed in range(self.num_of_cpu)]

        partial_results = [res.get() for res in mul_result]

        results = defaultdict(list)
        for row in partial_results:
            for card, info in row.items():
                results[card].extend(info)

        pool.close()

        min_score = sys.maxsize
        for card, info in results.items():
            total_size, total_score, total_num = 0, 0, 0

            for score, self_shoot_the_moon in info:
                total_score += score
                total_num += 1 if self_shoot_the_moon else 0
                total_size += 1

            mean_score = total_score / len(info)

            if mean_score < min_score:
                played_card = card
                min_score = mean_score

            ma, mi = -sys.maxsize, sys.maxsize
            for score, _ in info:
                if score > ma:
                    ma = score
                if score < mi:
                    mi = score

            self.say("simulate {} card with {:4d} times, and get {:.3f} score ({:.4f} ~ {:.4f}), num_moon={}, {:.2f}%", \
                card, total_size, mean_score, ma, mi, total_num, 100.0*total_num/total_size)

        self.say("pick {} card, cost {:.8} seconds", played_card, time.time()-stime)

        return played_card
