import os
import sys

import pickle

from intelligent_game import IntelligentGame
from intelligent_player import IntelligentPlayer

from nn import PolicyValueNet


def run(init_model, c_puct, time, n_games, filepath_out):
    data_buffer = []

    policy = PolicyValueNet(init_model)
    policy_value_fn = policy.predict

    players = [IntelligentPlayer(policy_value_fn, c_puct=c_puct, is_self_play=True, verbose=(True if player_idx == 3 else False)) for player_idx in range(4)]
    game = IntelligentGame(players, simulation_time_limit=time, verbose=True)

    for i in range(n_games):
        game.pass_cards(i%4)
        game.play()
        game.score()

        data_buffer.extend(game.get_memory())

        game.reset()

    policy.close()

    folder = os.path.dirname(filepath_out)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filepath_out, "wb") as out_file:
        pickle.dump(data_buffer, out_file)


if __name__ == "__main__":
    init_model, c_puct, time, n_games, filepath_out = \
        sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    print("init_model: {}, c_puct: {}, time: {}, n_game: {}, filepath_out: {}".format(\
        init_model, c_puct, time, n_games, filepath_out))

    run(init_model, c_puct, time, n_games, filepath_out)
