"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np
import multiprocessing as mp

from collections import defaultdict
from random import shuffle, choice

from card import Suit, Rank, Card, Deck
from rules import is_card_valid

from mcts_player import MCTSPlayer
from player import StupidPlayer

TIMEOUT_SECOND = 2
COUNT_CPU = mp.cpu_count()


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class AlphaPlayer(MCTSPlayer):
    def __init__(self, verbose=False):
        super(AlphaPlayer, self).__init__(verbose=verbose)

        self.C = 1.4
        self._is_selfplay = False


    def set_selfplay_mode(self, selfplay):
        self._is_selfplay = selfplay


    def play_card(self, hand_cards, game, simulation_time_limit=TIMEOUT_SECOND, temp_value=1e-3, return_prob=False):
        valid_cards = self.get_valid_cards(hand_cards, game.trick, game.trick_nr, game.is_heart_broken)

        played_card, played_prob = None, []
        if len(valid_cards) > 1:
            stime = time.time()

            plays, scores = defaultdict(int), defaultdict(int)
            pool = mp.Pool(processes=self.num_of_cpu)
            while time.time() - stime < simulation_time_limit:
                mul_result = [pool.apply_async(self.run_simulation, args=(game, plays, scores)) for _ in range(self.num_of_cpu)]
                results = [res.get() for res in mul_result]

                for tmp_plays, tmp_scores in results:
                    for k, v in tmp_plays.items():
                        plays[k] += v

                    for k, v in tmp_scores.items():
                        scores[k] += v

            pool.close()

            moves_states = []
            for card in valid_cards:
                seen_cards = [c for c in self.seen_cards[:]] + [card]
                moves_states.append((card, tuple(seen_cards)))

            n_visits = []
            average_score, played_card = sys.maxsize, None
            for card, states in moves_states:
                k = (game.current_player_idx, states)

                if k in plays:
                    avg_score = scores.get(k, 0) / plays[k]
                    if avg_score < average_score:
                        average_score = avg_score
                        played_card = card

                    self.say("{}, pick {}: score ---> {}/{}={:.2f}", valid_cards, card, scores[k], plays[k], avg_score)
                else:
                    self.say("not found {} in {}", k, plays)

                n_visits.append(plays.get(k, 0))

            played_prob = softmax(1.0/temp_value * np.log(np.array(n_visits) + 1e-10))

            self.say("(plays, scores, played_card, valid_cards) = ({}, {}, {}({:.2f}), {})",
                len(plays), len(scores), played_card, average_score, valid_cards)
        else:
            played_card = valid_cards[0]
            played_prob = [1.0]

            self.say("don't need simulation, can only play {} card", played_card)

        if self._is_selfplay:
            # add Dirichlet Noise for exploration (needed for
            # self-play training)
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(played_prob))))
                # update the root node and reuse the search tree
            #self.mcts.update_with_move(move)
        else:
            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob
            move = np.random.choice(valid_cards, p=played_prob)
            # reset the root node
            #self.mcts.update_with_move(-1)

        if return_prob:
            return move, list(zip(valid_cards, played_prob))
        else:
            return move
