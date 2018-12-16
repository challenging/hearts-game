import os

import numpy as np
import tensorflow as tf

from card import Suit, Rank
from card import Card, POINT_CARDS, CLUBS_T, SPADES_Q

from nn_utils import transform_game_info_to_nn, v2card


SORTED_CARDS = sorted(list(POINT_CARDS))
SORTED_CARDS = SORTED_CARDS[2:] + SORTED_CARDS[:2]

IS_DEBUG = False


class PolicyValueNet(object):
    def __init__(self, model_file=None):
        padding = "same"
        n_channel, n_suit, n_rank, n_player = 21, 4 ,13, 4

        activation_fn = tf.nn.relu

        # 1. inputs:
        self.inputs = tf.placeholder(tf.float32, shape=[None, n_channel, n_player, n_suit, n_rank], name="inputs")
        input_state = tf.transpose(self.inputs, [0, 2, 3, 4, 1], name="transpose_inputs")

        self.probs = tf.placeholder(tf.float32, shape=[None, 52], name="probs")

        self.score_1 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_2")
        self.score_2 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_3")
        self.score_3 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_4")
        self.score_4 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_5")
        self.score_5 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_6")
        self.score_6 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_7")
        self.score_7 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_8")
        self.score_8 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_9")
        self.score_9 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_10")
        self.score_10 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_11")
        self.score_11 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_12")
        self.score_12 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_13")
        self.score_13 = tf.placeholder(tf.int32, shape=[None, 4], name="hearts_ace")
        self.score_14 = tf.placeholder(tf.int32, shape=[None, 4], name="spades_queen")
        self.score_15 = tf.placeholder(tf.int32, shape=[None, 4], name="clubs_ten")

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        conv1 = tf.layers.conv3d(inputs=input_state,
                                 filters=32,
                                 kernel_size=[4, 4, 13],
                                 padding=padding,
                                 activation=activation_fn)

        conv2 = tf.layers.conv3d(inputs=conv1,
                                 filters=64,
                                 kernel_size=[4, 4, 13],
                                 padding=padding,
                                 activation=activation_fn)

        conv3 = tf.layers.conv3d(inputs=conv2,
                                 filters=128,
                                 kernel_size=[4, 4, 13],
                                 padding=padding,
                                 activation=activation_fn)

        # 3. Policy Networks
        action_conv = tf.layers.conv3d(inputs=conv3,
                                       filters=32,
                                       kernel_size=[1, 1, 1],
                                       padding=padding,
                                       activation=activation_fn)

        action_conv_flat = tf.reshape(action_conv, [-1, 32 * n_player * n_suit * n_rank])
        action_fc1 = tf.layers.dense(inputs=action_conv_flat, units=4096, activation=activation_fn)
        action_fc2 = tf.layers.dense(inputs=action_fc1, units=1024, activation=activation_fn)
        action_fc3 = tf.layers.dense(inputs=action_fc2, units=256, activation=activation_fn)
        self.action_fc = tf.layers.dense(inputs=action_fc3, units=52, activation=tf.nn.softmax)
        self.policy_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.probs, self.action_fc), 1))


        # 4. Value Networks
        evaluation_conv = tf.layers.conv3d(inputs=conv3,
                                           filters=4,
                                           kernel_size=[1, 1, 1],
                                           padding=padding,
                                           activation=activation_fn)

        evaluation_conv_flat = tf.reshape(evaluation_conv, [-1, 4 * n_player * n_suit * n_rank])
        evaluation_fc1 = tf.layers.dense(inputs=evaluation_conv_flat, units=128, activation=activation_fn)
        evaluation_fc2 = tf.layers.dense(inputs=evaluation_fc1, units=32, activation=activation_fn)

        def get_loss(labels, logits, name=None):
            return tf.losses.softmax_cross_entropy(labels, logits)

        self.score_evaluation_fc1 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_1 = get_loss(self.score_1, self.score_evaluation_fc1)

        self.score_evaluation_fc2 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_2 = get_loss(self.score_2, self.score_evaluation_fc2)

        self.score_evaluation_fc3 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_3 = get_loss(self.score_3, self.score_evaluation_fc3)

        self.score_evaluation_fc4 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_4 = get_loss(self.score_4, self.score_evaluation_fc4)

        self.score_evaluation_fc5 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_5 = get_loss(self.score_5, self.score_evaluation_fc5)

        self.score_evaluation_fc6 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_6 = get_loss(self.score_6, self.score_evaluation_fc6)

        self.score_evaluation_fc7 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_7 = get_loss(self.score_7, self.score_evaluation_fc7)

        self.score_evaluation_fc8 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_8 = get_loss(self.score_8, self.score_evaluation_fc8)

        self.score_evaluation_fc9 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_hearts_ten = get_loss(self.score_9, self.score_evaluation_fc9)

        self.score_evaluation_fc10 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_hearts_jack = get_loss(self.score_10, self.score_evaluation_fc10)

        self.score_evaluation_fc11 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_hearts_queen = get_loss(self.score_11, self.score_evaluation_fc11)

        self.score_evaluation_fc12 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_hearts_king = get_loss(self.score_12, self.score_evaluation_fc12)

        self.score_evaluation_fc13 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_hearts_ace = get_loss(self.score_13, self.score_evaluation_fc13)

        self.score_evaluation_fc14 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_clubs_ten = get_loss(self.score_14, self.score_evaluation_fc14)

        self.score_evaluation_fc15 = tf.layers.dense(inputs=evaluation_fc2, units=4, activation=tf.nn.tanh, reuse=False)
        self.loss_spades_queen = get_loss(self.score_15, self.score_evaluation_fc15)

        self.value_loss = self.loss_1 + self.loss_2 + self.loss_3 + self.loss_4 + self.loss_5 + \
                          self.loss_6 + self.loss_7 + self.loss_8 + \
                          self.loss_hearts_ten + self.loss_hearts_jack + self.loss_hearts_queen + \
                          self.loss_hearts_king + self.loss_spades_queen + self.loss_clubs_ten

        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])

        # 3-4 Add up to be the Loss function
        self.loss = l2_penalty + self.policy_loss + self.value_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.log(self.action_fc) * self.action_fc, 1)))

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
        if IS_DEBUG:
            print("     trick_cards:", np.array(trick_cards).shape)
            print("     score_cards:", np.array(score_cards).shape)
            print("  possible_cards:", np.array(possible_cards).shape)
            print("this_trick_cards:", np.array(this_trick_cards).shape)
            print("     valid_cards:", np.array(valid_cards).shape)
            print("   leading_cards:", np.array(leading_cards).shape)
            print("    expose_cards:", np.array(expose_cards).shape)

        inputs = np.concatenate((trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards), \
                                axis=1)

        if IS_DEBUG:
            print("shape of inputs: {}".format(inputs.shape))

        results = self.session.run([self.action_fc, self.score_evaluation_fc1, self.score_evaluation_fc2,
                                    self.score_evaluation_fc3, self.score_evaluation_fc4, self.score_evaluation_fc5,
                                    self.score_evaluation_fc6, self.score_evaluation_fc7, self.score_evaluation_fc8,
                                    self.score_evaluation_fc9, self.score_evaluation_fc10, self.score_evaluation_fc11,
                                    self.score_evaluation_fc12, self.score_evaluation_fc13, self.score_evaluation_fc14,
                                    self.score_evaluation_fc15],
                                   feed_dict={self.inputs: inputs})
        return results


    def transform_results(self, all_player_idx, all_valid_cards, all_expose_cards, all_results, is_need_card=True):
        global SORTED_CARDS

        probs_batch, score_cards_batch = [], []
        for current_player_idx, valid_cards, expose_cards, results in zip(all_player_idx, all_valid_cards, all_expose_cards, all_results):
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

            probs = []
            if is_need_card:
                where = np.where(valid_cards[current_player_idx] == 1)
                for pos_x, pos_y in zip(where[0], where[1]):
                    probs.append(((pos_x, 1<<pos_y), results[0][0][pos_x*13+pos_y]))
                    #probs.append(((pos_x, 1<<pos_y), np.exp(results[0][0][pos_x*13+pos_y])))

                #print("results", results)
                #print("-->", valid_cards[current_player_idx])
                #print("probs", probs)

            else:
                probs = []
                where = np.where(valid_cards[0, current_player_idx] != 99999999)
                for pos_x, pos_y in zip(where[0], where[1]):
                    probs.append(results[0][pos_x*13+pos_y])
                    #probs.append(np.exp(results[0][pos_x*13+pos_y]))

            probs_batch.append(probs)
            score_cards_batch.append(scores)

        #print("      probs_batch:", probs_batch)
        #print("score_cards_batch:", score_cards_batch)
        return probs_batch, score_cards_batch


    def predict(self, trick_nr, state):
        trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards = \
            transform_game_info_to_nn(state, trick_nr)

        results = self.policy_value(np.array([trick_cards]), np.array([score_cards]), np.array([possible_cards]), \
                                    np.array([this_trick_cards]), np.array([valid_cards]), \
                                    np.array([leading_cards]), np.array([expose_cards]))

        all_results = [[]]
        for idx in range(16):
            all_results[-1].append(results[idx])

        probs, scores = self.transform_results([state.start_pos], valid_cards, expose_cards, all_results)

        return probs[0], scores[0]


    def policy_value_fn(self, current_player_idx, trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards):
        results = self.policy_value(trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards)

        all_results = []
        for idx in range(len(trick_cards)):
            all_results.append([])

            for sub_idx in range(16):
                all_results[-1].append(results[sub_idx][idx])

        return self.transform_results(current_player_idx, valid_cards, expose_cards, all_results, is_need_card=False)


    def get_card_owner(self, trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards, scores):
        inputs = np.concatenate((trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards), \
                                axis=1)

        scores = np.array(scores)

        card_owner = self.session.run([self.score_evaluation_fc1, self.score_evaluation_fc2, self.score_evaluation_fc3,
                                       self.score_evaluation_fc4, self.score_evaluation_fc5, self.score_evaluation_fc6,
                                       self.score_evaluation_fc7, self.score_evaluation_fc8, self.score_evaluation_fc9,
                                       self.score_evaluation_fc10, self.score_evaluation_fc11, self.score_evaluation_fc12,
                                       self.score_evaluation_fc13, self.score_evaluation_fc14, self.score_evaluation_fc15],
                                       feed_dict={self.inputs: inputs,
                                                  self.score_1: scores[:,0],
                                                  self.score_2: scores[:,1],
                                                  self.score_3: scores[:,2],
                                                  self.score_4: scores[:,3],
                                                  self.score_5: scores[:,4],
                                                  self.score_6: scores[:,5],
                                                  self.score_7: scores[:,6],
                                                  self.score_8: scores[:,7],
                                                  self.score_9: scores[:,8],
                                                  self.score_10: scores[:,9],
                                                  self.score_11: scores[:,10],
                                                  self.score_12: scores[:,11],
                                                  self.score_13: scores[:,12],
                                                  self.score_14: scores[:,13],
                                                  self.score_15: scores[:,14],})

        return card_owner


    def train_step(self, \
                   trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards, \
                   probs, scores, learning_rate):
        inputs = np.concatenate((trick_cards, score_cards, possible_cards, this_trick_cards, valid_cards, leading_cards, expose_cards), \
                                axis=1)

        scores = np.array(scores)

        loss, policy_loss, value_loss, entropy, _ = self.session.run(
                [self.loss, self.policy_loss, self.value_loss, self.entropy, self.optimizer],
                feed_dict={self.inputs: inputs,
                           self.probs: probs,
                           self.score_1: scores[:,0],
                           self.score_2: scores[:,1],
                           self.score_3: scores[:,2],
                           self.score_4: scores[:,3],
                           self.score_5: scores[:,4],
                           self.score_6: scores[:,5],
                           self.score_7: scores[:,6],
                           self.score_8: scores[:,7],
                           self.score_9: scores[:,8],
                           self.score_10: scores[:,9],
                           self.score_11: scores[:,10],
                           self.score_12: scores[:,11],
                           self.score_13: scores[:,12],
                           self.score_14: scores[:,13],
                           self.score_15: scores[:,14],
                           self.learning_rate: learning_rate})

        return loss, policy_loss, value_loss, entropy


    def save_model(self, model_path):
        self.saver.save(self.session, model_path)
        print("save model in {}".format(model_path))


    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
        print("restore model from {}".format(model_path))


    def close(self):
        self.session.close()
