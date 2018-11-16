import os
import sys

import time
import pickle

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

BASEPATH = "prob"
BASEPATH_MODEL = os.path.join(BASEPATH, "model")
BASEPATH_BEST_MODEL = os.path.join(BASEPATH_MODEL, "best")
BASEPATH_DATA = os.path.join(BASEPATH, "data")
BASEPATH_LOG = os.path.join("log", "intelligent_player")

#policy = PolicyValueNet()
#policy_value_fn = policy.predict


class TrainPipeline():
    def __init__(self, init_model=None):
        self.policy = PolicyValueNet(init_model)
        self.policy_value_fn = self.policy.predict

        # training params
        self.learn_rate = 2e-6
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.c_puct = 2**10

        self.buffer_size = 2**16
        self.batch_size = 32  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.play_batch_size = 16
        self.epochs = 64  # num of train_steps for each update
        self.check_freq = 4
        self.kl_targ = 0.02

        self.best_score = sys.maxsize
        self.pure_mcts_simulation_time_limit = 1
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy

        players = [IntelligentPlayer(self.policy_value_fn, c_puct=self.c_puct, is_self_play=True, verbose=(True if player_idx>-1 else False)) for player_idx in range(4)]

        self.game = Game(players, simulation_time_limit=2.5, verbose=True)


    def collect_selfplay_data(self, n_games, game_idx):
        """collect self-play data for training"""

        filepath_out = os.path.join(BASEPATH_LOG, "game.{}.log".format(game_idx))
        self.game.out_file = open(filepath_out, "w")

        for i in range(n_games):
            stime = time.time()

            self.game.pass_cards(i%4)
            self.game.play()
            self.game.score()

            play_data = self.game.get_memory()
            self.data_buffer.extend(play_data)
            self.episode_len = len(play_data)

            self.game.reset()

            print("cost {:.4f} seconds, training game: {:3d}".format(time.time()-stime, i+1))

        filepath_in = os.path.join(BASEPATH_DATA, str(time.time()*1000)+".pkl")
        with open(filepath_in, "wb") as in_file:
            pickle.dump(self.data_buffer, in_file)

        self.game.out_file.close()


    def policy_update(self):
        for i in range(self.epochs):
            remaining_batch, trick_batch = [], []
            must_batch_1, must_batch_2, must_batch_3, must_batch_4 = [], [], [], []
            scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4 = [], [], [], []
            hand_batch, valid_batch, expose_batch = [], [], []
            probs_batch, scores_batch = [], []

            for remaining, trick, must, scards, hand_cards, valid_cards, expose_info, probs, scores in sample(self.data_buffer, self.batch_size):
                remaining_batch.append(full_cards(remaining))
                trick_batch.append(limit_cards(trick, 3))

                must_batch_1.append(limit_cards(must[0], 4))
                must_batch_2.append(limit_cards(must[1], 4))
                must_batch_3.append(limit_cards(must[2], 4))
                must_batch_4.append(limit_cards(must[3], 4))

                scards_batch_1.append(limit_cards(scards[0], 15))
                scards_batch_2.append(limit_cards(scards[1], 15))
                scards_batch_3.append(limit_cards(scards[2], 15))
                scards_batch_4.append(limit_cards(scards[3], 15))

                hand_batch.append(limit_cards(hand_cards, 13))

                valid_batch.append(limit_cards(valid_cards, 13))

                expose_batch.append(expose_info)
                probs_batch.append(limit_cards(dict(zip(valid_cards, probs)), 13))

                scores_batch.append(scores)

            old_probs, old_v = self.policy.policy_value(\
                remaining_batch, trick_batch, \
                must_batch_1, must_batch_2, must_batch_3, must_batch_4, \
                scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4, \
                hand_batch, valid_batch, expose_batch)

            loss, policy_loss, value_loss = self.policy.train_step(
                    remaining_batch, trick_batch, \
                    must_batch_1, must_batch_2, must_batch_3, must_batch_4, \
                    scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4, \
                    hand_batch, valid_batch, expose_batch, \
                    probs_batch, scores_batch,\
                    self.learn_rate*self.lr_multiplier)

            print("epoch: {:3d}/{:3d}, policy_loss: {:.8f}, value_loss: {:.8f}, loss: {:.8f}".format(\
                i+1, self.epochs, policy_loss, value_loss, loss))

            new_probs, new_v = self.policy.policy_value(\
                remaining_batch, trick_batch, \
                must_batch_1, must_batch_2, must_batch_3, must_batch_4, \
                scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4, \
                hand_batch, valid_batch, expose_batch)

            #kl = np.mean(np.sum(old_probs * (
            #        np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
            #        axis=1))

            #if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
            #    break

        # adaptively adjust the learning rate
        """
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        """

        print(("learning_rate: {:.8f}, lr_multiplier:{:.8f}").format(self.learn_rate, self.lr_multiplier))

        return loss, policy_loss, value_loss


    def policy_evaluate(self, game_idx, n_games=1):
        filepath_in = os.path.join(BASEPATH_LOG, "evaluation.{}.log".format(game_idx))
        out_file = open(filepath_in, "w")

        current_mcts_player = IntelligentPlayer(self.policy_value_fn, self.c_puct, is_self_play=False, verbose=True)
        players = [NewSimplePlayer(verbose=False) for _ in range(3)] + [current_mcts_player]

        setting_cards = read_card_games("game/game_0008/02/game_1541503661.pkl")
        final_scores, proactive_moon_scores, shooting_moon_scores = \
            evaluate_players(n_games, players, setting_cards, verbose=True, out_file=out_file)

        myself_score = np.mean(final_scores[3])
        others_score = np.mean(final_scores[:3])

        print("myself_score: {:.4f}, others_score: {:.4f}".format(myself_score, others_score))

        out_file.close()

        return myself_score/others_score


    def run(self, game_batch_num):
        try:
            for i in range(game_batch_num):
                self.collect_selfplay_data(self.play_batch_size, i+1)

                print("batch i: {}, episode_len: {}, memory_size: {}".format(\
                    i+1, self.episode_len, len(self.data_buffer)))

                #for played_data in self.data_buffer:
                #    print_a_memory(played_data)

                if len(self.data_buffer) >= self.batch_size:
                    loss, policy_loss, value_loss = self.policy_update()

                if i%self.check_freq == 0:
                    score = self.policy_evaluate(i+1)

                    print("current self-play batch: {}, and score ratio: {:.4f}".format(i+1, score))

                    #self.policy.save_model(os.path.join(BASEPATH_MODEL, 'current_policy.model'))
                    if score < self.best_score:
                        print("New best policy!!!!!!!!", score, self.best_score)
                        self.best_score = score

                        # update the best_policy
                        #policy.save_model(os.path.join(BASEPATH_BEST_MODEL, 'best_policy.model'))
                        if score < 1:
                            self.pure_mcts_simulation_time_limit <<= 1
        except KeyboardInterrupt:
            print('\nquit')


if __name__ == "__main__":
    num_of_games = int(sys.argv[1])
    sub_folder = sys.argv[2]
    model_filepath = sys.argv[3] if len(sys.argv) > 3 else None

    BASEPATH = "prob"
    BASEPATH_MODEL = os.path.join(BASEPATH, "model_{}".format(sub_folder))
    BASEPATH_BEST_MODEL = os.path.join(BASEPATH_MODEL, "best")
    BASEPATH_DATA = os.path.join(BASEPATH, "data_{}".format(sub_folder))
    BASEPATH_LOG = os.path.join("log", "intelligent_player_{}".format(sub_folder))

    for folder in [BASEPATH_MODEL, BASEPATH_DATA, BASEPATH_LOG]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    training_pipeline = TrainPipeline(model_filepath)
    training_pipeline.run(max(num_of_games, 1))
