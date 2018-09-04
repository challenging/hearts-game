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
from game import Game

from mcts_player import MCTSPlayer
from player import SimplePlayer

from nn_utils import card2v

TIMEOUT_SECOND = 0.9
COUNT_CPU = 1#mp.cpu_count()


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class AlphaPlayer(MCTSPlayer):
    def __init__(self, policy, verbose=False):
        super(AlphaPlayer, self).__init__(verbose=verbose)

        self.C = 1.4
        self._is_selfplay = False
        self.policy = policy


    def set_selfplay(self, selfplay):
        self._is_selfplay = selfplay


    def play_card(self, hand_cards, game, simulation_time_limit=TIMEOUT_SECOND, temp_value=1e-3, return_prob=False):
        game.are_hearts_broken()
        valid_cards = self.get_valid_cards(hand_cards, game)

        played_card, played_prob = None, []
        plays, scores = defaultdict(int), defaultdict(int)
        if len(valid_cards) > 1:
            stime = time.time()

            local_game = Game([SimplePlayer(verbose=False) for idx in range(4)], verbose=False)
            for player in local_game.players:
                player.seen_cards = copy.deepcopy(self.seen_cards)
            local_game.trick = game.trick[:]
            local_game.trick_nr = game.trick_nr
            local_game.current_player_idx = game.current_player_idx

            local_game.take_pig_card = game.take_pig_card
            local_game.is_heart_broken = game.is_heart_broken
            local_game.is_shootmoon = game.is_shootmoon

            local_game._player_hands = game._player_hands[:]
            local_game._cards_taken = game._cards_taken[:]

            cards = [card2v(card) for card in valid_cards] + [0 for _ in range(12-len(valid_cards))]

            action_probs, network_value = self.policy([game.current_status()], [cards])
            print(cards, network_value.shape)

            pool = mp.Pool(processes=self.num_of_cpu)
            while time.time() - stime < simulation_time_limit:
                #mul_result = [pool.apply_async(self.run_simulation, args=(local_game, plays, scores)) for _ in range(self.num_of_cpu)]
                #results = [res.get() for res in mul_result]
                results = [self.run_simulation(copy.deepcopy(local_game), plays, scores, network_value)]

                for tmp_plays, tmp_scores in results:
                    for k, v in tmp_plays.items():
                        plays[k] += v

                    for k, v in tmp_scores.items():
                        scores[k] += v
            pool.close()

            n_visits = []
            for card in valid_cards:
                state = (card, tuple([c for c in self.seen_cards[:]] + [card]))
                n_visits.append(plays.get(state, 0))

            #played_prob = softmax(1.0/temp_value * np.log(np.array(n_visits) + 1e-10))
            n_visits = np.array(n_visits)
            played_prob = n_visits / n_visits.sum()
        else:
            played_prob = np.array([1.0])

        if self._is_selfplay:
            move = np.random.choice(valid_cards, p=0.75*played_prob + 0.25*np.random.dirichlet(0.3*np.ones(len(played_prob))))

            self.say("(plays, valid_card, played_card) = ({}, {}, {}({:.2f}))",
                len(plays), list(zip(valid_cards, played_prob)), move)
        else:
            move = np.random.choice(valid_cards, p=played_prob)

            self.say("(prob: {}, played_card)", list(zip(valid_cards, played_prob)), move)

        if return_prob:
            return move, list(zip(valid_cards, played_prob))
        else:
            return move


    def run_simulation(self, game, plays, scores, network_value=0):
        hand_cards = game._player_hands[game.current_player_idx]
        remaining_cards = self.get_remaining_cards(hand_cards)
        self.redistribute_cards(game, remaining_cards[:])

        trick_nr = game.trick_nr
        tmp_plays, tmp_scores = {}, {}

        player = game.players[self.position]
        valid_cards = player.get_valid_cards(game._player_hands[self.position], game)

        played_card, is_all_pass, total_visits = None, True, 0
        moves_states = []
        for card in valid_cards:
            seen_cards = [c for c in player.seen_cards[:]] + [card]

            key = (card, tuple(seen_cards))
            moves_states.append(key)

            is_all_pass &= (key in plays)
            if is_all_pass:
                total_visits += plays[key]

        if is_all_pass:
            value, state = max(
                (network_value - (scores[state] / plays[state]) + self.C * np.sqrt(np.log(total_visits) / plays[state]), state) for state in moves_states)

            played_card = state[0]
        else:
            played_card = choice(valid_cards)

        state = (played_card, tuple(game.players[0].seen_cards[:]) + (played_card,))

        game.step(played_card)
        for _ in range(4-len(game.trick)):
            game.step()

        for _ in range(13-game.trick_nr):
            game.play_trick()

        if trick_nr < 6:
            self.overwrite_game_score_func(game)

        game.score()

        tmp_plays[state] = 1
        tmp_scores[state] = self.score_func(game.player_scores)

        return tmp_plays, tmp_scores
