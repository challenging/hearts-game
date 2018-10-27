# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku
@author: Junxiao Song
"""

from __future__ import print_function

import sys

import numpy as np

from random import sample
from collections import deque

from card import Suit, Rank, Card
from card import read_card_games
from rules import evaluate_players

from intelligent_game import IntelligentGame as Game
from intelligent_player import IntelligentPlayer
from new_simple_player import NewSimplePlayer
from new_simulated_player import MonteCarloPlayer7

from nn import PolicyValueNet
from nn_utils import card2v, v2card, full_cards, limit_cards, print_a_memory

#from mcts import policy_value_fn
#policy = policy_value_fn

policy = PolicyValueNet()
policy_value_fn = policy.predict


class TrainPipeline():
    def __init__(self, init_model=None):
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.c_puct = 2

        self.buffer_size = 2**15
        self.batch_size = 2**9  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.play_batch_size = 32
        self.epochs = 4  # num of train_steps for each update
        self.check_freq = 2
        self.kl_targ = 0.02

        self.best_score = sys.maxsize
        self.pure_mcts_simulation_time_limit = 1
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy

        players = [IntelligentPlayer(policy_value_fn, c_puct=self.c_puct, is_self_play=True, verbose=False) for player_idx in range(4)]

        self.game = Game(players, verbose=True)


    def collect_selfplay_data(self, n_games):
        """collect self-play data for training"""
        for i in range(n_games):
            self.game.pass_cards(i%4)
            self.game.play()
            self.game.score()

            play_data = self.game.get_memory()
            self.data_buffer.extend(play_data)
            self.episode_len = len(play_data)

            self.game.reset()


    def policy_update(self):
        for i in range(self.epochs):
            remaining_batch, trick_batch = [], []
            must_batch_1, must_batch_2, must_batch_3, must_batch_4 = [], [], [], []
            scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4 = [], [], [], []
            valid_batch = []
            probs_batch, scores_batch = [], []

            for remaining, trick, must, scards, valid_cards, played_cards, probs, scores in sample(self.data_buffer, self.batch_size):
                remaining_batch.append(full_cards(remaining))
                trick_batch.append(limit_cards(trick, 3))

                must_batch_1.append(limit_cards(must[0], 4))
                must_batch_2.append(limit_cards(must[1], 4))
                must_batch_3.append(limit_cards(must[2], 4))
                must_batch_4.append(limit_cards(must[3], 4))

                scards_batch_1.append(full_cards(scards[0]))
                scards_batch_2.append(full_cards(scards[1]))
                scards_batch_3.append(full_cards(scards[2]))
                scards_batch_4.append(full_cards(scards[3]))

                valid_batch.append(full_cards(valid_cards))

                probs_batch.append(full_cards(dict(zip(played_cards, probs))))

                scores_batch.append(scores)

            old_probs, old_v = policy.policy_value(\
                remaining_batch, trick_batch, \
                must_batch_1, must_batch_2, must_batch_3, must_batch_4, \
                scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4, \
                valid_batch)

            loss, policy_loss, value_loss = policy.train_step(
                    remaining_batch, trick_batch, \
                    must_batch_1, must_batch_2, must_batch_3, must_batch_4, \
                    scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4, \
                    valid_batch, probs_batch, scores_batch,\
                    self.learn_rate*self.lr_multiplier)

            new_probs, new_v = policy.policy_value(\
                remaining_batch, trick_batch, \
                must_batch_1, must_batch_2, must_batch_3, must_batch_4, \
                scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4, \
                valid_batch)

            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1))

            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(("kl:{:.5f}, lr_multiplier:{:.3f}, loss:{},").format(kl, self.lr_multiplier, loss))

        return loss, policy_loss, value_loss


    def policy_evaluate(self, n_games=1):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = IntelligentPlayer(policy_value_fn, self.c_puct, is_self_play=False, verbose=True)
        players = [NewSimplePlayer(verbose=False) for _ in range(3)] + [current_mcts_player]

        setting_cards = read_card_games("game/game_0008/game_1534672484.pkl")
        final_scores, proactive_moon_scores, shooting_moon_scores = \
            evaluate_players(n_games, players, setting_cards, verbose=True)

        myself_score = np.mean(final_scores[3])
        others_score = np.mean(final_scores[:3])

        print("myself_score: {:.4f}, others_score: {:.4f}".format(myself_score, others_score))

        return myself_score/others_score


    def run(self, game_batch_num):
        """run the training pipeline"""
        try:
            for i in range(game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)

                print("batch i: {}, episode_len: {}, memory_size: {}".format(\
                    i+1, self.episode_len, len(self.data_buffer)))

                #for played_data in self.data_buffer:
                #    print_a_memory(played_data)

                if len(self.data_buffer) >= self.batch_size:
                    loss, policy_loss, value_loss = self.policy_update()

                if i%self.check_freq == 0:
                    score = self.policy_evaluate()

                    print("current self-play batch: {}, and score ratio: {:.4f}".format(i+1, score))

                    policy.save_model('model/current_policy.model')
                    if score < self.best_score:
                        print("New best policy!!!!!!!!", score, self.best_score)
                        self.best_score = score

                        # update the best_policy
                        policy.save_model('model/best_policy.model')
                        if i > 0: self.pure_mcts_simulation_time_limit <<= 1
        except KeyboardInterrupt:
            print('\nquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run(max(sys.argv[1], 1))
