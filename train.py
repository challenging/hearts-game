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

from nn import CNNPolicyValueNet as Net
from nn_utils import card2v, v2card, full_cards, limit_cards, print_a_memory


DEBUG = False


def run(idx, init_model, c_puct, time, n_games, filepath_out, filepath_log):
    command_line = "./make_memory.py {} {} {} {} {}".format(\
        init_model, c_puct, time, n_games, filepath_out, filepath_log)
    print("cmd: {}".format(command_line))

    args = shlex.split(command_line)

    with open(filepath_log, "w") as in_file:
        subprocess.check_call(args, stdout=sys.stdout if DEBUG else in_file)


class TrainPipeline():
    def __init__(self, init_model=None, card_time=0.2):
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
        self.learning_rate = 2e-6
        self.c_puct = 256

        self.buffer_size = 2**16
        self.batch_size = 16
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.cpu_count = min(mp.cpu_count(), 12)
        self.epochs = 32

        self.play_batch_size = int(self.batch_size*self.epochs/52/self.cpu_count)
        print("cpu_count={}, batch_size={}, epochs={}, play_batch_size={}".format(\
            self.cpu_count, self.batch_size, self.epochs, self.play_batch_size))

        self.c_puct_evaluation = self.c_puct/2
        self.filepath_evaluation = os.path.join("game", "game_0002", "01", "game_*.pkl")

        self.card_time = card_time
        self.pure_mcts_simulation_time_limit = 1


    def collect_selfplay_data(self, n_games):
        self.data_buffer.clear()
        #print("clear the self.data_buffer({})".format(len(self.data_buffer)))

        row_number = int(time.time())
        filepath_data = os.path.join(self.basepath_data, "{}.{}.pkl".format(row_number, "{:02d}"))
        filepath_log = os.path.join(self.basepath_log, "{}.{}.log".format(row_number, "{:02d}"))

        for filepath in [filepath_data, filepath_log]:
            folder = os.path.dirname(filepath_log)
            if os.path.exists(folder):
                shutil.rmtree(folder)

            os.makedirs(folder)

        pool = mp.Pool(processes=self.cpu_count)
        mul_result = [pool.apply_async(run, 
                                       args=(idx,
                                             self.init_model, 
                                             self.c_puct, 
                                             self.card_time, 
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
        for i in range(self.epochs):
            remaining_batch = []
            trick_nr_batch, trick_order_batch, pos_batch, played_order_batch, trick_batch = [], [], [], [], []
            must_batch, score_batch, historical_batch, hand_batch, valid_batch = [], [], [], [], []
            expose_batch, void_batch, winning_batch = [], [], []
            probs_batch, scores_batch = [], []

            samples = sample(self.data_buffer, self.batch_size)
            for remaining, trick_nr, trick_order, pos, played_order, trick, must, history, scards, hand, valid, expose_info, void_info, winning_info, probs, scores in samples:
                remaining_batch.append(full_cards(remaining))

                trick_nr_batch.append([trick_nr])
                trick_order_batch.append(trick_order)
                pos_batch.append([pos])
                played_order_batch.append([played_order])
                trick_batch.append(limit_cards(trick, 3))

                must_batch.append([limit_cards(must[0], 4), limit_cards(must[1], 4), limit_cards(must[2], 4), limit_cards(must[3], 4)])
                historical_batch.append([limit_cards(history[0], 13), limit_cards(history[1], 13), limit_cards(history[2], 13), limit_cards(history[3], 13)])
                score_batch.append([limit_cards(scards[0], 15), limit_cards(scards[1], 15), limit_cards(scards[2], 15), limit_cards(scards[3], 15)])
                hand_batch.append(limit_cards(hand, 13))
                valid_batch.append(limit_cards(valid, 13))

                expose_batch.append(expose_info)
                void_batch.append(void_info)
                winning_batch.append(winning_info)

                probs_batch.append(limit_cards(dict(zip(valid, probs)), 13))
                scores_batch.append(scores)

            loss, policy_loss, value_loss = self.policy.train_step(
                    remaining_batch, \
                    trick_nr_batch, trick_order_batch, pos_batch, played_order_batch, trick_batch, \
                    must_batch, historical_batch, score_batch, hand_batch, valid_batch, \
                    expose_batch, void_batch, winning_batch, \
                    probs_batch, scores_batch,\
                    self.learning_rate)

            print("epoch: {:3d}/{:3d}, policy_loss: {:.8f}, value_loss: {:.8f}, loss: {:.8f}".format(\
                i+1, self.epochs, policy_loss, value_loss, loss))

        return loss, policy_loss, value_loss


    def policy_evaluate(self, n_games=1):
        filepath_in = os.path.join(self.basepath_log, "evaluation.log")
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
            evaluate_players(n_games, players, setting_cards, verbose=True, out_file=out_file)

        myself_score, others_score = np.mean(final_scores[3]), np.mean(final_scores[:3])
        print("myself_score: {:.4f}, others_score: {:.4f}".format(myself_score, others_score))

        if not DEBUG:
            out_file.close()

        return myself_score, others_score


    def run(self, start_idx=0):
        try:
            best_score = sys.maxsize

            for i in range(start_idx, sys.maxsize):
                basepath_round = self.basepath_round.format(i+1)
                self.basepath_data = os.path.join(basepath_round, "data")
                self.basepath_model = os.path.join(basepath_round, "model")
                self.basepath_log = os.path.join(basepath_round, "log")

                if i == -1:
                    myself_score, others_score = self.policy_evaluate()
                    print("current self-play batch: {}, and myself_score: {:.2f}, others_score: {:.2f}".format(\
                        i+1, myself_score, others_score))

                    continue

                self.collect_selfplay_data(self.play_batch_size)
                print("batch i: {}, memory_size: {}".format(i+1, len(self.data_buffer)))

                if DEBUG:
                    for played_data in self.data_buffer:
                        print_a_memory(played_data)

                loss, policy_loss, value_loss = self.policy_update()

                myself_score, others_score = self.policy_evaluate()
                print("current self-play batch: {}, and myself_score: {:.2f}, others_score: {:.2f}".format(\
                    i+1, myself_score, others_score))

                filepath_model = os.path.join(self.basepath_model, "current_policy.model")
                folder = os.path.dirname(filepath_model)
                if not os.path.exists(folder):
                    os.makedirs(folder)

                self.policy.save_model(filepath_model)

                if myself_score <= best_score:
                    if best_score != sys.maxsize:
                        filepath_model = os.path.join(self.basepath, "best_policy.model")
                        self.policy.save_model(filepath_model)

                        self.init_model = filepath_model

                    best_score = myself_score

                    if myself_score/others_score < 1:
                        self.pure_mcts_simulation_time_limit <<= 1
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
        model_filepath = os.path.join("prob", "best_policy.model")

    simulated_time = float(sys.argv[1])
    round_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print("set model_filepath={}, round_idx={}".format(model_filepath, round_idx))

    training_pipeline = TrainPipeline(model_filepath, simulated_time)
    training_pipeline.run(round_idx)
