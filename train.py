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

from nn import PolicyValueNet as Net
from nn_utils import print_a_memory


DEBUG = False


def run(idx, init_model, c_puct, time, min_times, n_games, filepath_out, filepath_log):
    command_line = "./make_memory.py {} {} {} {} {} {}".format(\
        init_model, c_puct, time, min_times, n_games, filepath_out, filepath_log)
    print("cmd: {}".format(command_line))

    args = shlex.split(command_line)

    with open(filepath_log, "w") as in_file:
        subprocess.check_call(args, stdout=sys.stdout if DEBUG else in_file)


class TrainPipeline():
    def __init__(self, init_model=None, card_time=2):
        self.basepath = "prob"
        self.basepath_round = os.path.join(self.basepath, "round_{:04d}")
        self.basepath_data = os.path.join(self.basepath_round, "data")
        self.basepath_model = os.path.join(self.basepath_round, "model")
        self.basepath_log = os.path.join(self.basepath_round, "log")

        if not os.path.exists(self.basepath):
            os.makedirs(self.basepath)

        self.init_model = init_model

        self.init_nn_model()

        # training params
        self.learning_rate = 1e-4
        self.c_puct = 1024
        self.min_times = 32

        self.buffer_size = 2**16
        self.batch_size = 8
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.cpu_count = min(mp.cpu_count(), 12)

        self.play_batch_size = 4
        self.epochs = int(52*self.cpu_count/8/self.play_batch_size)
        print("cpu_count={}, batch_size={}, epochs={}, play_batch_size={}".format(\
            self.cpu_count, self.batch_size, self.epochs, self.play_batch_size))

        self.c_puct_evaluation = self.c_puct
        self.filepath_evaluation = os.path.join("game", "game_0008", "01", "game_*.pkl")
        print("filepath_evaluation={}".format(self.filepath_evaluation))

        self.card_time = card_time
        self.pure_mcts_simulation_time_limit = 1


    def init_nn_model(self):
        self.policy = Net(self.init_model)
        self.policy_value_fn = self.policy.predict

        if self.init_model is None:
            filepath_model = os.path.join(self.basepath, "init_policy.model")
            self.policy.save_model(filepath_model)

            self.init_model = filepath_model
            print("create init model in {}".format(filepath_model))


    def collect_selfplay_data(self, n_games):
        self.data_buffer.clear()

        row_number = int(time.time())
        filepath_data = os.path.join(self.basepath_data, "{}.{}.pkl".format(row_number, "{:02d}"))
        filepath_log = os.path.join(self.basepath_log, "{}.{}.log".format(row_number, "{:02d}"))

        for filepath in [filepath_data, filepath_log]:
            folder = os.path.dirname(filepath_log)
            if not os.path.exists(folder):
                os.makedirs(folder)

        pool = mp.Pool(processes=self.cpu_count)
        mul_result = [pool.apply_async(run, 
                                       args=(idx,
                                             self.init_model, 
                                             self.c_puct, 
                                             self.card_time, 
                                             self.min_times,
                                             n_games, 
                                             filepath_data.format(idx), 
                                             filepath_log.format(idx))) for idx in range(self.cpu_count)]
        results = [res.get() for res in mul_result]

        for filepath_in in glob.glob(os.path.join(self.basepath_data, "*.pkl")):
            with open(filepath_in, "rb") as in_file:
                data_buffer = pickle.load(in_file)
                print("collect {} memory from {}".format(len(data_buffer), filepath_in))

                self.data_buffer.extend(data_buffer)

        pool.close()

        print("collect {} memory for training".format(len(self.data_buffer)))


    def policy_update(self):
        n_epochs = max(256, len(self.data_buffer) // self.batch_size * 2)

        for i in range(n_epochs):
            trick_cards_batch, score_cards_batch, possible_cards = [], [], []
            this_trick_batch, valid_cards_batch = [], []
            leading_cards_batch, expose_cards_batch = [], []
            probs_batch, scores_batch = [], []

            samples = sample(self.data_buffer, self.batch_size)
            for current_player_idx, trick_cards, score_cards, possible_cards, this_trick, valid_cards, is_leading, is_expose, probs, scores in samples:
                trick_cards_batch.append(transform_trick_cards(trick_cards))
                score_cards_batch.append(transform_score_cards(score_cards))
                possible_cards_batch.append(transform_possible_cards(possible_cards))
                this_trick_batch.append(transform_this_trick_cards(current_player_idx, this_trick_cards))
                valid_cards_batch.append(transform_valid_cards(current_player_idx, valid_cards))
                leading_cards_batch.append(transform_leading_cards(current_player_idx, leading_cards))
                expose_cards_batch.append(transform_expose_cards(expose_cards))

                probs_batch.append(probs)
                scores_batch.append(scores)

            loss, policy_loss, value_loss = self.policy.train_step(
                    trick_cards_batch, score_cards_batch, possible_cards, \
                    this_trick_batch, valid_cards_batch, \
                    leading_cards_batch, expose_cards_batch, \
                    probs_batch, scores_batch,\
                    self.learning_rate)

            print("epoch: {:4d}/{:4d}, policy_loss: {:.8f}, value_loss: {:.8f}, loss: {:.8f}".format(\
                i+1, n_epochs, policy_loss, value_loss, loss))


    def policy_evaluate(self, n_games=1):
        filepath_in = os.path.join(self.basepath_log, "evaluation.{}.log".format(int(time.time())))
        folder = os.path.dirname(filepath_in)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if DEBUG:
            out_file = sys.stdout
        else:
            out_file = open(filepath_in, "w")

        current_mcts_player = IntelligentPlayer(self.policy_value_fn, self.c_puct_evaluation, is_self_play=False, verbose=True)
        players = [NewSimplePlayer(verbose=False) for _ in range(3)] + [current_mcts_player]

        setting_cards = read_card_games(self.filepath_evaluation)

        final_scores, proactive_moon_scores, shooting_moon_scores = \
            evaluate_players(n_games, players, setting_cards, verbose=True, is_rotating=False, out_file=out_file)

        myself_score, others_score = np.mean(final_scores[3]), np.mean(final_scores[:3])
        print("myself_score: {:.4f}, others_score: {:.4f}".format(myself_score, others_score))

        if not DEBUG:
            out_file.close()

        return myself_score, others_score


    def run(self, start_idx=0):
        try:
            best_score = sys.maxsize

            start_idx
            while True:
                basepath_round = self.basepath_round.format(start_idx+1)
                self.basepath_data = os.path.join(basepath_round, "data")
                self.basepath_model = os.path.join(basepath_round, "model")
                self.basepath_log = os.path.join(basepath_round, "log")

                self.init_nn_model()

                if start_idx == 0:
                    myself_score, others_score = self.policy_evaluate()
                    print("current self-play batch: {}, and myself_score: {:.2f}, others_score: {:.2f}".format(\
                        start_idx+1, myself_score, others_score))

                    sys.exit(0)
                else:
                    self.collect_selfplay_data(self.play_batch_size)
                    print("batch i: {}, memory_size: {}".format(start_idx+1, len(self.data_buffer)))

                    if DEBUG:
                        for played_data in self.data_buffer:
                            print_a_memory(played_data)

                    self.policy_update()

                    myself_score, others_score = self.policy_evaluate()
                    print("current self-play batch: {}, and myself_score: {:.4f}, others_score: {:.4f}".format(\
                        start_idx+1, myself_score, others_score))

                    filepath_model = os.path.join(self.basepath, "round{:04d}_policy.model".format(start_idx+1))
                    folder = os.path.dirname(filepath_model)
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    self.policy.save_model(filepath_model)

                    if myself_score <= best_score:
                        filepath_model = os.path.join(self.basepath, "best_policy.model")
                        self.policy.save_model(filepath_model)

                        self.init_model = filepath_model

                        best_score = myself_score
                        if myself_score/others_score < 1:
                            self.pure_mcts_simulation_time_limit <<= 1

                        self.card_time = 1
                    else:
                        self.card_time *= 1.1
                        #self.c_puct *= 1.1
                        #self.min_times *= 1.1

                        start_idx -= 1
                start_idx += 1
        except KeyboardInterrupt:
            print('\nquit')


if __name__ == "__main__":
    model_filepath = os.path.join("prob", "best_policy.model.meta")
    if not os.path.exists(model_filepath):
        model_filepath = None

        model_filepath = os.path.join("prob", "init_policy.model.meta")
        if not os.path.exists(model_filepath):
            model_filepath = None
        else:
            model_filepath = os.path.join("prob", "init_policy.model")
    else:
        model_filepath = os.path.join("prob", "round0001_policy.model")

    simulated_time = float(sys.argv[1])
    round_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print("set model_filepath={}, round_idx={}".format(model_filepath, round_idx))

    training_pipeline = TrainPipeline(model_filepath, simulated_time)
    training_pipeline.run(round_idx)
