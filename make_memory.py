#!/usr/bin/env python

import os
os.environ["CUDA_VISABLE_DEVICES"] = ""

import sys

import pickle

from intelligent_game import IntelligentGame
from intelligent_mcts import IntelligentMCTS
from intelligent_player import IntelligentPlayer

from nn import PolicyValueNet as Net


def run(init_model, c_puct, time, min_times, n_games, filepath_out):
    data_buffer = []

    policy = Net(init_model)
    mcts = IntelligentMCTS(policy.predict, None, c_puct, min_times=min_times)

    players = [IntelligentPlayer(policy.predict,
                                 c_puct=c_puct,
                                 mcts=mcts,
                                 is_self_play=True,
                                 min_times=min_times,
                                 verbose=(True if player_idx == 3 else True)) for player_idx in range(4)]

    game = IntelligentGame(players, simulation_time_limit=time, verbose=True)

    count_s, count_f = 0, 0
    for i in range(n_games):
        try:
            game.pass_cards(i%4)
            game.play()
            game.score()

            data_buffer.extend(game.get_memory())

            game.reset()

            count_s += 1
        except Exception as e:
            game = IntelligentGame(players, simulation_time_limit=time, verbose=True)
            game.reset()

            count_f += 1

            raise

    sys.stderr.write("count_s, count_f = {}, {}\n".format(count_s, count_f))
    policy.close()

    folder = os.path.dirname(filepath_out)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filepath_out, "wb") as out_file:
        pickle.dump(data_buffer, out_file)


if __name__ == "__main__":
    print(sys.argv)
    init_model, c_puct, time, min_times, n_games, filepath_out = \
        sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]

    print("init_model: {}, c_puct: {}, time: {}, min_times:{}, n_game: {}, filepath_out: {}".format(\
        init_model, c_puct, time, min_times, n_games, filepath_out))

    run(init_model, c_puct, time, min_times, n_games, filepath_out)

    sys.exit(0)
