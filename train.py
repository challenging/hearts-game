import os
import sys

import copy
import glob
import shutil

import shlex
import subprocess

import time
import pickle

import numpy as np
import multiprocessing as mp

from random import sample
from collections import deque

from card import Suit, Rank, Card
from card import read_card_games
from rules import evaluate_players

from intelligent_game import IntelligentGame as Game
from intelligent_player import IntelligentPlayer
from new_simple_player import NewSimplePlayer

from nn import PolicyValueNet
from cnn import CNNPolicyValueNet as Net
from nn_utils import card2v, v2card, full_cards, limit_cards, print_a_memory


def run(idx, init_model, c_puct, time, n_games, filepath_out, filepath_log):
    command_line = "python make_memory.py {} {} {} {} {}".format(\
        init_model, c_puct, time, n_games, filepath_out, filepath_log)
    print("cmd: {}".format(command_line))

    args = shlex.split(command_line)

    with open(filepath_log, "w") as in_file:
        p = subprocess.Popen(args, stdout=in_file)
        ret = p.communicate()

    return ret


class TrainPipeline():
    def __init__(self, init_model=None, card_time=0.2, play_batch_size=2):
        self.basepath = "prob"
        self.basepath_round = os.path.join(self.basepath, "round_{:04d}")
        self.basepath_data = os.path.join(self.basepath_round, "data")
        self.basepath_model = os.path.join(self.basepath_round, "model")
        self.basepath_log = os.path.join(self.basepath_round, "log")

        if not os.path.exists(self.basepath):
            os.makedirs(self.basepath)

        self.init_model = init_model

        self.policy = Net(self.init_model)
        self.policy_value_fn = self.policy.predict

        if self.init_model is None:
            filepath_model = os.path.join(self.basepath, "init_policy.model")
            self.policy.save_model(filepath_model)

            self.init_model = filepath_model
            print("create init model in {}".format(filepath_model))

        # training params
        self.learn_rate = 2e-6
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.c_puct = 1600

        self.buffer_size = 2**16
        self.batch_size = 32  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.play_batch_size = play_batch_size
        self.epochs = 256  # num of train_steps for each update
        self.check_freq = 1
        self.kl_targ = 0.02

        self.best_score = sys.maxsize

        self.card_time = card_time
        self.pure_mcts_simulation_time_limit = 1


    def collect_selfplay_data(self, n_games):
        self.data_buffer.clear()
        print("clear the self.data_buffer({})".format(len(self.data_buffer)))

        row_number = int(time.time())
        filepath_data = os.path.join(self.basepath_data, "{}.{}.pkl".format(row_number, "{:02d}"))
        filepath_log = os.path.join(self.basepath_log, "{}.{}.log".format(row_number, "{:02d}"))

        for filepath in [filepath_data, filepath_log]:
            folder = os.path.dirname(filepath_log)
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)

        cpu_count = mp.cpu_count()-2

        pool = mp.Pool(processes=cpu_count)
        mul_result = [pool.apply_async(run, 
                                       args=(idx,
                                             self.init_model, 
                                             self.c_puct, 
                                             self.card_time, 
                                             n_games, 
                                             filepath_data.format(idx), 
                                             filepath_log.format(idx))) for idx in range(cpu_count)]
        results = [res.get() for res in mul_result]

        for filepath_in in glob.glob(os.path.join(self.basepath_data, "*.pkl")):
            with open(filepath_in, "rb") as in_file:
                data_buffer = pickle.load(in_file)
                print("collect {} memory from {}".format(len(data_buffer), filepath_in))

                self.data_buffer.extend(data_buffer)

        print("collect {} memory for training".format(len(self.data_buffer)))


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

            """
            old_probs, old_v = self.policy.policy_value(\
                remaining_batch, trick_batch, \
                must_batch_1, must_batch_2, must_batch_3, must_batch_4, \
                scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4, \
                hand_batch, valid_batch, expose_batch)
            """

            loss, policy_loss, value_loss = self.policy.train_step(
                    remaining_batch, trick_batch, \
                    must_batch_1, must_batch_2, must_batch_3, must_batch_4, \
                    scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4, \
                    hand_batch, valid_batch, expose_batch, \
                    probs_batch, scores_batch,\
                    self.learn_rate*self.lr_multiplier)

            print("epoch: {:3d}/{:3d}, policy_loss: {:.8f}, value_loss: {:.8f}, loss: {:.8f}".format(\
                i+1, self.epochs, policy_loss, value_loss, loss))

            """
            new_probs, new_v = self.policy.policy_value(\
                remaining_batch, trick_batch, \
                must_batch_1, must_batch_2, must_batch_3, must_batch_4, \
                scards_batch_1, scards_batch_2, scards_batch_3, scards_batch_4, \
                hand_batch, valid_batch, expose_batch)

            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1))

            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
            """

        """
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        """

        print("kl: {:8f}, learning_rate: {:.8f}, lr_multiplier:{:.8f}".format(\
            0, self.learn_rate, self.lr_multiplier))

        return loss, policy_loss, value_loss


    def policy_evaluate(self, n_games=1):
        filepath_in = os.path.join(self.basepath_log, "evaluation.log")
        folder = os.path.dirname(filepath_in)
        if not os.path.exists(folder):
            os.makedirs(folder)

        out_file = open(filepath_in, "w")

        current_mcts_player = IntelligentPlayer(self.policy_value_fn, self.c_puct, is_self_play=False, verbose=True)
        players = [NewSimplePlayer(verbose=False) for _ in range(3)] + [current_mcts_player]

        setting_cards = read_card_games("game/game_0004/02/game_1541503661.pkl")
        #setting_cards = read_card_games("game/game_0001/01/game_1542415443.pkl")

        final_scores, proactive_moon_scores, shooting_moon_scores = \
            evaluate_players(n_games, players, setting_cards, verbose=True, out_file=out_file)

        myself_score = np.mean(final_scores[3])
        others_score = np.mean(final_scores[:3])

        print("myself_score: {:.4f}, others_score: {:.4f}".format(myself_score, others_score))

        out_file.close()

        return myself_score, others_score


    def run(self, game_batch_num):
        try:
            for i in range(game_batch_num):
                basepath_round = self.basepath_round.format(i+1)
                self.basepath_data = os.path.join(basepath_round, "data")
                self.basepath_model = os.path.join(basepath_round, "model")
                self.basepath_log = os.path.join(basepath_round, "log")

                self.collect_selfplay_data(self.play_batch_size)
                print("batch i: {}, memory_size: {}".format(i+1, len(self.data_buffer)))

                #for played_data in self.data_buffer:
                #    print_a_memory(played_data)

                if len(self.data_buffer) >= self.batch_size:
                    loss, policy_loss, value_loss = self.policy_update()

                if i%self.check_freq == 0:
                    myself_score, others_score = self.policy_evaluate()
                    print("current self-play batch: {}, and myself_score: {:.2f}, others_score: {:.2f}".format(\
                        i+1, myself_score, others_score))

                    filepath_model = os.path.join(self.basepath_model, "current_policy.model")
                    folder = os.path.dirname(filepath_model)
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    self.init_model = filepath_model

                    self.policy.save_model(filepath_model)
                    if myself_score <= self.best_score:
                        self.best_score = myself_score

                        filepath_model = os.path.join(self.basepath_model, "best_policy.model")

                        self.policy.save_model(filepath_model)
                        if myself_score/others_score < 1: self.pure_mcts_simulation_time_limit <<= 1
        except KeyboardInterrupt:
            print('\nquit')


if __name__ == "__main__":
    num_of_games = int(sys.argv[1])
    simulated_time = float(sys.argv[2])
    played_batch = int(sys.argv[3])
    model_filepath = sys.argv[4] if len(sys.argv) > 4 else None

    training_pipeline = TrainPipeline(model_filepath, simulated_time, played_batch)
    training_pipeline.run(max(num_of_games, 1))
