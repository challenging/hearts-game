import os

import numpy as np
import tensorflow as tf

from card import Suit, Rank
from card import Card, POINT_CARDS, CLUBS_T, SPADES_Q

from nn_utils import transform_game_info_to_nn, v2card


SORTED_CARDS = sorted(list(POINT_CARDS))
SORTED_CARDS = SORTED_CARDS[2:] + SORTED_CARDS[:2]


class PolicyValueNet(object):
    def __init__(self, model_file=None):
        n_channel, n_suit, n_rank, n_player = 21, 4 ,13, 4

        activation_fn = tf.nn.relu

        # 1. inputs:
        self.inputs = tf.placeholder(tf.float32, shape=[None, n_channel, n_player, n_suit, n_rank], name="inputs")
        input_state = tf.transpose(self.inputs, [0, 2, 3, 4, 1], name="transpose_inputs")

        self.probs = tf.placeholder(tf.float32, shape=[None, 52], name="probs")

        self.score_1 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_2")
        self.score_2 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_3")
        self.score_3 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_4")
        self.score_4 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_5")
        self.score_5 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_6")
        self.score_6 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_7")
        self.score_7 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_8")
        self.score_8 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_9")
        self.score_9 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_10")
        self.score_10 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_11")
        self.score_11 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_12")
        self.score_12 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_13")
        self.score_13 = tf.placeholder(tf.float32, shape=[None, 4], name="hears_ace")
        self.score_14 = tf.placeholder(tf.float32, shape=[None, 4], name="spades_queen")
        self.score_15 = tf.placeholder(tf.float32, shape=[None, 4], name="clubs_ten")

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        conv1 = tf.layers.conv3d(inputs=input_state,
                                 filters=32,
                                 kernel_size=[4, 4, 4],
                                 padding="same",
                                 activation=activation_fn)

        conv2 = tf.layers.conv3d(inputs=conv1,
                                 filters=64,
                                 kernel_size=[4, 4, 4],
                                 padding="same",
                                 activation=activation_fn)

        conv3 = tf.layers.conv3d(inputs=conv2,
                                 filters=128,
                                 kernel_size=[4, 4, 4],
                                 padding="same",
                                 activation=activation_fn)

        # 3. Policy Networks
        action_conv = tf.layers.conv3d(inputs=conv3,
                                       filters=16,
                                       kernel_size=[1, 1, 1],
                                       padding="same",
                                       activation=activation_fn)

        action_conv_flat = tf.reshape(action_conv, [-1, 16 * n_player * n_suit * n_rank])
        action_fc1 = tf.layers.dense(inputs=action_conv_flat, units=128, activation=activation_fn)
        self.action_fc = tf.layers.dense(inputs=action_fc1, units=52, activation=tf.nn.log_softmax)
        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.probs, self.action_fc), 1)))


        # 4. Value Networks
        evaluation_conv = tf.layers.conv3d(inputs=conv3,
                                           filters=4,
                                           kernel_size=[1, 1, 1],
                                           padding="same",
                                           activation=activation_fn)

        evaluation_conv_flat = tf.reshape(evaluation_conv, [-1, 4 * n_player * n_suit * n_rank])
        evaluation_fc1 = tf.layers.dense(inputs=evaluation_conv_flat, units=128, activation=activation_fn)
        evaluation_fc2 = tf.layers.dense(inputs=evaluation_fc1, units=32, activation=activation_fn)

        def get_loss(labels, logits):
            return tf.losses.softmax_cross_entropy(labels, logits)

        self.score_evaluation_fc1 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_1 = get_loss(self.score_1, self.score_evaluation_fc1)

        self.score_evaluation_fc2 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_2 = get_loss(self.score_2, self.score_evaluation_fc2)

        self.score_evaluation_fc3 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_3 = get_loss(self.score_3, self.score_evaluation_fc3)

        self.score_evaluation_fc4 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_4 = get_loss(self.score_4, self.score_evaluation_fc4)

        self.score_evaluation_fc5 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_5 = get_loss(self.score_5, self.score_evaluation_fc5)

        self.score_evaluation_fc6 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_6 = get_loss(self.score_6, self.score_evaluation_fc6)

        self.score_evaluation_fc7 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_7 = get_loss(self.score_7, self.score_evaluation_fc7)

        self.score_evaluation_fc8 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_8 = get_loss(self.score_8, self.score_evaluation_fc8)

        self.score_evaluation_fc9 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_9 = get_loss(self.score_9, self.score_evaluation_fc9)

        self.score_evaluation_fc10 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_10 = get_loss(self.score_10, self.score_evaluation_fc10)

        self.score_evaluation_fc11 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_11 = get_loss(self.score_11, self.score_evaluation_fc11)

        self.score_evaluation_fc12 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_12 = get_loss(self.score_12, self.score_evaluation_fc12)

        self.score_evaluation_fc13 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_13 = get_loss(self.score_13, self.score_evaluation_fc13)

        self.score_evaluation_fc14 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_14 = get_loss(self.score_14, self.score_evaluation_fc14)

        self.score_evaluation_fc15 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh)
        loss_15 = get_loss(self.score_15, self.score_evaluation_fc15)

        self.score_hearts_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7 + \
                                 loss_8 + loss_9 + loss_10 + loss_11 + loss_12 + loss_13

        self.score_spades_loss = loss_14
        self.score_clubs_loss = loss_15

        self.value_loss = self.score_hearts_loss + self.score_spades_loss + self.score_clubs_loss

        self.loss = self.policy_loss + self.value_loss

        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = l2_penalty + self.policy_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Make a session
        self.session = tf.Session()

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None and os.path.exists(model_file):
            print("start to restore model from {}".format(model_file),)
            self.restore_model(model_file)
            print("done")


    def policy_value(self, trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards):
        global SORTED_CARDS

        inputs = np.concatenate((trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards), \
                                axis=0)

        results = self.session.run([self.action_fc, self.score_evaluation_fc1, self.score_evaluation_fc2,
                                    self.score_evaluation_fc3, self.score_evaluation_fc4, self.score_evaluation_fc5,
                                    self.score_evaluation_fc6, self.score_evaluation_fc7, self.score_evaluation_fc8,
                                    self.score_evaluation_fc9, self.score_evaluation_fc10, self.score_evaluation_fc11,
                                    self.score_evaluation_fc12, self.score_evaluation_fc13, self.score_evaluation_fc14,
                                    self.score_evaluation_fc15],
                                   feed_dict={self.inputs: [inputs]})
        return results


    def predict(self, trick_nr, state):
        trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards = \
            transform_game_info_to_nn(state, trick_nr)

        results = self.policy_value(np.array(trick_cards), np.array(score_cards), np.array(possible_cards), \
                                    np.array(this_trick_cards), np.array(valid_cards), \
                                    np.array(leading_cards), np.array(expose_cards))

        is_expose = (np.max(expose_cards) == 2)

        scores, double_player_idx = [0, 0, 0, 0], None
        for card, sub_results in zip(SORTED_CARDS, results[1:]):
            player_idx = np.argmax(sub_results)

            if card.suit == Suit.hearts:
                scores[player_idx] += (2 if is_expose else 1)
            elif card == CLUBS_T:
                double_player_idx = player_idx
            elif card == SPADES_Q:
                scores[player_idx] += 13

        scores[double_player_idx] <<= 1

        where, probs = np.where(valid_cards[0, state.start_pos] == 1), []
        #print("where", where, valid_cards, results)

        for pos_x, pos_y in zip(where[0], where[1]):
            pos = pos_x*13+pos_y
            probs.append(((pos_x, 1<<pos_y), np.exp(results[0][0][pos])))

        return probs, scores

    """
    def policy_value_fn(self, trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards):
        return self.policy_value(trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards)
    """


    def train_step(self, \
                   trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards, \
                   probs, scores, learning_rate):

        inputs = np.concatenate((trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards), \
                                axis=0)

        loss, policy_loss, value_loss, _ = self.session.run(
                [self.loss, self.policy_loss, self.value_loss, self.optimizer],
                feed_dict={self.inputs: inputs,
                           self.probs: probs,
                           self.score_1: scores[0],
                           self.score_2: scores[1],
                           self.score_3: scores[2],
                           self.score_4: scores[3],
                           self.score_5: scores[4],
                           self.score_6: scores[5],
                           self.score_7: scores[6],
                           self.score_8: scores[7],
                           self.score_9: scores[8],
                           self.score_10: scores[9],
                           self.score_11: scores[10],
                           self.score_12: scores[11],
                           self.score_13: scores[12],
                           self.score_14: scores[13],
                           self.score_15: scores[14],
                           self.learning_rate: learning_rate})

        return loss, policy_loss, value_loss


    def save_model(self, model_path):
        self.saver.save(self.session, model_path)
        print("save model in {}".format(model_path))


    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)


    def close(self):
        self.session.close()
