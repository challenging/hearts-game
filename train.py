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

from card import Suit, Rank, Card, POINT_CARDS
from card import read_card_games
from rules import evaluate_players

from intelligent_game import IntelligentGame as Game
from intelligent_player import IntelligentPlayer
from new_simple_player import NewSimplePlayer

from nn import PolicyValueNet as Net
from nn_utils import transform_trick_cards, transform_score_cards, transform_possible_cards, transform_results
from nn_utils import transform_this_trick_cards, transform_valid_cards, transform_expose_cards, transform_leading_cards
from nn_utils import print_a_memory

SORTED_CARDS = sorted(list(POINT_CARDS))
SORTED_CARDS = SORTED_CARDS[2:] + SORTED_CARDS[:2]

MIN_CARD_TIME = 0.8

DEBUG = False


def run(idx, init_model, c_puct, time, min_times, n_games, filepath_out, filepath_log):
    command_line = "./make_memory.py {} {} {} {} {} {}".format(\
        init_model, c_puct, time, min_times, n_games, filepath_out, filepath_log)
    print("cmd: {}".format(command_line))

    args = shlex.split(command_line)

    with open(filepath_log, "w") as in_file:
        try:
            subprocess.check_call(args, stdout=sys.stdout if DEBUG else in_file)
        except Exception as e:
            import traceback

            traceback.print_exc()


class TrainPipeline():
    def __init__(self, init_model=None, card_time=2, n_played_game=1):
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
        self.kl_targ = 0.02
        self.lr_multiplier = 1.0
        self.learning_rate = 1e-4

        self.c_puct = 2**11
        self.min_times = 2**8

        self.buffer_size = 2**16
        self.batch_size = 32
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.cpu_count = min(mp.cpu_count(), 12)

        self.play_batch_size = n_played_game
        self.epochs = int(52*self.cpu_count*self.play_batch_size/self.batch_size)*2
        print("cpu_count={}, batch_size={}, epochs={}, play_batch_size={}, min_times={}".format(\
            self.cpu_count, self.batch_size, self.epochs, self.play_batch_size, self.min_times))

        self.c_puct_evaluation = self.c_puct
        self.filepath_evaluation = os.path.join("game", "game_0004", "01", "game_*.pkl")
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
        global MIN_CARD_TIME

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
                                             max(MIN_CARD_TIME, self.card_time), 
                                             self.min_times,
                                             n_games, 
                                             filepath_data.format(idx), 
                                             filepath_log.format(idx))) for idx in range(self.cpu_count)]
        results = [res.get() for res in mul_result]

        pool.close()


    def collect_memory(self):
        self.data_buffer.clear()

        for filepath_in in glob.glob(os.path.join(self.basepath_data, "*.pkl")):
            with open(filepath_in, "rb") as in_file:
                data_buffer = pickle.load(in_file)
                self.data_buffer.extend(data_buffer)

        print("collect {} memory for training".format(len(self.data_buffer)))


    def policy_update(self):
        for i in range(self.epochs):
            player_idx_batch = []
            trick_cards_batch, score_cards_batch, possible_cards_batch = [], [], []
            this_trick_batch, valid_cards_batch = [], []
            leading_cards_batch, expose_cards_batch = [], []
            probs_batch, scores_batch, scores_cards_batch = [], [], []

            samples = sample(self.data_buffer, self.batch_size)
            for current_player_idx, trick_cards, score_cards, possible_cards, this_trick, valid_cards, is_leading, is_expose, probs, scores in samples:
                player_idx_batch.append(current_player_idx)

                trick_cards_batch.append(transform_trick_cards(trick_cards))
                score_cards_batch.append(transform_score_cards(score_cards))
                possible_cards_batch.append(transform_possible_cards(possible_cards))
                this_trick_batch.append(transform_this_trick_cards(current_player_idx, this_trick))
                valid_cards_batch.append(transform_valid_cards(current_player_idx, valid_cards))
                leading_cards_batch.append(transform_leading_cards(current_player_idx, is_leading))
                expose_cards_batch.append(transform_expose_cards(is_expose))

                probs_batch.append(probs)
                scores_batch.append(transform_results(scores))
                scores_cards_batch.append(scores)

            old_probs, old_values = self.policy.policy_value_fn( \
                player_idx_batch, \
                trick_cards_batch, score_cards_batch, possible_cards_batch, \
                this_trick_batch, valid_cards_batch, \
                leading_cards_batch, expose_cards_batch)

            old_card_owners = self.policy.get_card_owner( \
                trick_cards_batch, score_cards_batch, possible_cards_batch, \
                this_trick_batch, valid_cards_batch, \
                leading_cards_batch, expose_cards_batch, scores_batch)

            # update the PolicyValueNetwork
            loss, policy_loss, value_loss, entropy = self.policy.train_step(
                trick_cards_batch, score_cards_batch, possible_cards_batch, \
                this_trick_batch, valid_cards_batch, \
                leading_cards_batch, expose_cards_batch, \
                probs_batch, scores_batch,\
                self.learning_rate*self.lr_multiplier)

            new_probs, new_values = self.policy.policy_value_fn( \
                player_idx_batch, \
                trick_cards_batch, score_cards_batch, possible_cards_batch, \
                this_trick_batch, valid_cards_batch, \
                leading_cards_batch, expose_cards_batch)

            new_card_owners = self.policy.get_card_owner( \
                trick_cards_batch, score_cards_batch, possible_cards_batch, \
                this_trick_batch, valid_cards_batch, \
                leading_cards_batch, expose_cards_batch, scores_batch)

            kl = []
            for old_prob, new_prob in zip(old_probs, new_probs):
                old_prob = np.array(old_prob)
                new_prob = np.array(new_prob)

                kl.append(np.sum(old_prob * (np.log(old_prob+1e-16) - np.log(new_prob+1e-16))))

            kl = np.mean(kl)

            if kl > self.kl_targ*4:
                print("early stopping because of {:.8f} > {:.8f}".format(kl, self.kl_targ*4))
                break

            print("epoch: {:4d}/{:4d}, kl: {:.8f}, policy_loss: {:.8f}, value_loss: {:.8f}, loss: {:.8f}, entropy: {:.8f}".format(\
                i+1, self.epochs, kl, policy_loss, value_loss, loss, entropy))


            old_card_owner = []
            for idx in range(len(old_card_owners)):
                old_card_owner.append([])

                for sub_idx in range(15):
                    old_card_owner[-1].append(np.argmax(old_card_owners[sub_idx][idx]))

            new_card_owner = []
            for idx in range(len(new_card_owners)):
                new_card_owner.append([])

                for sub_idx in range(15):
                    new_card_owner[-1].append(np.argmax(new_card_owners[sub_idx][idx]))

            owners = {}
            for card in SORTED_CARDS:
                owners[card] = [0, 0]

            for old_cards, new_cards, owner_cards in zip(old_card_owner, new_card_owner, scores_cards_batch):
                for card, old_card, new_card in zip(SORTED_CARDS, old_cards, new_cards):
                    for real_player_idx, owner_card in enumerate(owner_cards):
                        if card in owner_card:
                            break

                    old_player_idx, new_player_idx = np.argmax(old_card), np.argmax(new_card)
                    if old_player_idx == real_player_idx:
                        owners[card][0] += 1

                    if new_player_idx == real_player_idx:
                        owners[card][1] += 1

            for card, losses in owners.items():
                old_mean_loss = losses[0] / len(scores_cards_batch)
                new_mean_loss = losses[1] / len(scores_cards_batch)
                print("\t loss of {}: {:.4f} --> {:.4f}".format(card, old_mean_loss, new_mean_loss))

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            print("decrease the self.lr_multiplier from {} --> ".format(self.lr_multiplier),)
            self.lr_multiplier /= 1.5

            print(self.lr_multiplier)
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            print("increase the self.lr_multiplier from {} --> ".format(self.lr_multiplier),)
            self.lr_multiplier *= 1.5

            print(self.lr_multiplier)


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
                else:
                    self.collect_selfplay_data(self.play_batch_size)
                    self.collect_memory()
                    print("batch i: {:04d}, memory_size: {:5d}".format(start_idx+1, len(self.data_buffer)))

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

                        best_score = myself_score
                        if myself_score/others_score < 1:
                            self.pure_mcts_simulation_time_limit <<= 1

                        self.card_time = 1
                    else:
                        self.card_time *= 1.1

                    self.init_model = filepath_model

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
    n_played_game = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    print("set model_filepath={}, round_idx={}, n_played_game={}".format(\
        model_filepath, round_idx, n_played_game))

    training_pipeline = TrainPipeline(model_filepath, simulated_time, n_played_game)
    training_pipeline.run(round_idx)
