# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku
@author: Junxiao Song
"""

from __future__ import print_function

import sys
import copy
import random

import numpy as np

from pprint import pprint
from collections import defaultdict, deque

from card import Suit, Rank, Card
from card import read_card_games
from rules import evaluate_players

from alpha_game import AlphaGame as Game
from alpha_player import AlphaPlayer
from simulated_player import MonteCarloPlayer4 as MonteCarloPlayer

#from policy_value_net_tensorflow import PolicyValueNet


class TrainPipeline():
    def __init__(self, init_model=None):
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 512  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 2**15
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 4  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 2
        self.best_score = sys.maxsize
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy

        """
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        """

        players = [AlphaPlayer(policy_value_network=None, verbose=False) for _ in range(4)]
        for player in players:
            player.set_selfplay(True)

        self.game = Game(players, verbose=False)


    def collect_selfplay_data(self, n_games):
        """collect self-play data for training"""
        for i in range(n_games):
            self.game.pass_cards()
            self.game.play()
            self.game.score()

            play_data = self.game.get_memory()
            self.data_buffer.extend(copy.deepcopy(play_data))
            self.episode_len = len(play_data)

            self.game.reset()


    def policy_update(self):
        """update the policy-value net"""
        states_batch, cards_batch, probs_batch, scores_batch = [], [], [], []
        for states, cards, probs, scores in random.sample(self.data_buffer, self.batch_size):
            states_batch.append(states)
            cards_batch.append(cards)
            probs_batch.append(probs)
            scores_batch.append(scores)

        #old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    states_batch,
                    cards_batch,
                    probs_batch,
                    scores_batch,
                    self.learn_rate)#*self.lr_multiplier)

            """
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )

            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
            """

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))

        return loss, entropy


    def policy_evaluate(self, n_games=8):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = AlphaPlayer(policy_value_network=None)
        players = [current_mcts_player] + [MonteCarloPlayer(verbose=False) for _ in range(3)]

        setting_cards = read_card_games("game/game_0008/game_1534672484.pkl")
        statss = evaluate_players(n_games, players, setting_cards, verbose=False)

        min_score, second_score = None, None
        for idx, score in enumerate(sorted([stats.mean for stats in statss])):
            if idx == 0:
                min_score = score
            elif idx == 1:
                second_score = score
                break

        return np.mean(statss[0])


    def run(self, game_batch_num):
        """run the training pipeline"""
        try:
            for i in range(game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)

                print("batch i: {}, episode_len: {}, memory_size: {}".format(\
                    i+1, self.episode_len, len(self.data_buffer)))

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                if (i+1) % self.check_freq == 0:
                    score = self.policy_evaluate()

                    print("current self-play batch: {}".format(i+1))

                    #self.policy_value_net.save_model('./current_policy.model')
                    if score < self.best_score:
                        print("New best policy!!!!!!!!", score, self.best_score)
                        self.best_score = score

                        # update the best_policy
                        """
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                        """
        except KeyboardInterrupt:
            print('\nquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run(4)
