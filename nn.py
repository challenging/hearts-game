import os

import numpy as np
import tensorflow as tf

from nn_utils import transform_game_info_to_nn


class CNNPolicyValueNet(object):
    def __init__(self, model_file=None, size_embed=8):
        size_score_card = 15
        activation_fn = tf.nn.relu

        # 1. input:
        self.remaining_cards = tf.placeholder(tf.int32, shape=[None, 52], name="remaining_cards")
        self.trick_nr = tf.placeholder(tf.int32, shape=[None, 1], name="trick_nr")
        #self.trick_order = tf.placeholder(tf.int32, shape=[None, 4], name="trick_order")
        #self.position = tf.placeholder(tf.int32, shape=[None, 1], name="position")
        self.played_order = tf.placeholder(tf.int32, shape=[None, 1], name="played_order")
        self.trick_cards = tf.placeholder(tf.int32, shape=[None, 3], name="trick_cards")
        self.must_cards = tf.placeholder(tf.int32, shape=[None, 4, 4], name="must_cards")
        self.historical_cards = tf.placeholder(tf.int32, shape=[None, 4, 13], name="historical_cards")
        self.score_cards = tf.placeholder(tf.int32, shape=[None, 4, size_score_card], name="score_cards")
        self.hand_cards = tf.placeholder(tf.int32, shape=[None, 13], name="hand_cards")
        self.valid_cards = tf.placeholder(tf.int32, shape=[None, 13], name="valid_cards")
        self.expose_info = tf.placeholder(tf.float32, shape=[None, 4], name="expose_info")
        self.void_info = tf.placeholder(tf.float32, shape=[None, 4, 4], name="void_info")
        self.winning_info = tf.placeholder(tf.float32, shape=[None, 4, 13], name="winning_info")
        self.probs = tf.placeholder(tf.float32, shape=[None, 13],   name="probs")
        self.score = tf.placeholder(tf.float32, shape=[None, 4], name="score")

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        def conv2d(name, input, size):
            conv1 = tf.layers.conv2d(inputs=input,
                                     filters=8,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     data_format="channels_last",
                                     activation=activation_fn)

            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=16,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     data_format="channels_last",
                                     activation=activation_fn)

            conv = tf.layers.conv2d(inputs=conv2,
                                    filters=4,
                                    kernel_size=[1, 1],
                                    padding="same",
                                    data_format="channels_last",
                                    activation=activation_fn)

            return tf.reshape(conv, [-1, int(4*size*size_embed)], name="reshape_{}".format(name))


        def conv2d_2(name, input, size):
            conv1 = tf.layers.conv2d(inputs=input,
                                     filters=8,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     data_format="channels_last",
                                     activation=activation_fn)

            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=16,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     data_format="channels_last",
                                     activation=activation_fn)

            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=32,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     data_format="channels_last",
                                     activation=activation_fn)

            conv = tf.layers.conv2d(inputs=conv3,
                                    filters=4,
                                    kernel_size=[1, 1],
                                    padding="same",
                                    data_format="channels_last",
                                    activation=activation_fn)

            return tf.reshape(conv, [-1, 4*4*size], name="reshape_{}".format(name))

        def conv1d(name, input, size):
            conv1 = tf.layers.conv1d(inputs=input,
                                     filters=8,
                                     kernel_size=3,
                                     padding="same",
                                     activation=activation_fn)

            conv2 = tf.layers.conv1d(inputs=conv1,
                                     filters=16,
                                     kernel_size=3,
                                     padding="same",
                                     activation=activation_fn)

            conv = tf.layers.conv1d(inputs=conv2,
                                    filters=4,
                                    kernel_size=3,
                                    padding="same",
                                    activation=activation_fn)

            return tf.reshape(conv, [-1, 4*size], name="reshape_{}".format(name))


        # Define the embedding layer
        nr_embeddings = tf.Variable(tf.random_uniform([13, 4], -1.0, 1.0), name="embedding_nr")
        cards_embeddings = tf.Variable(tf.random_uniform([53, size_embed], -1.0, 1.0), name="embedding_cards")
        #position_embeddings = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0), name="embedding_position")
        step_embeddings = tf.Variable(tf.random_uniform([52, 4], -1.0, 1.0), name="embedding_step")

        remaining_embed = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.remaining_cards), [-1, 52, size_embed, 1], name="reshape_remaining_embed")
        remaining_flat = conv2d_2("remaining", remaining_embed, int(size_embed/4)*52)

        trick_nr = tf.reshape(tf.nn.embedding_lookup(nr_embeddings, self.trick_nr), [-1, 4], name="reshape_trick_nr")
        #trick_order = tf.reshape(tf.nn.embedding_lookup(position_embeddings, self.trick_order), [-1, 4*4], name="reshape_trick_nr")

        trick_embed = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.trick_cards), [-1, 3, size_embed, 1])
        trick_flat = conv2d("trick", trick_embed, 3)

        #position = tf.reshape(tf.nn.embedding_lookup(position_embeddings, self.position), [-1, 4], name="reshape_trick_nr")
        step = tf.reshape(tf.nn.embedding_lookup(step_embeddings, self.played_order), [-1, 4], name="reshape_trick_nr")

        must_flat = conv2d("must", tf.nn.embedding_lookup(cards_embeddings, self.must_cards), 16/size_embed)

        historical_flat = conv2d_2("history", tf.nn.embedding_lookup(cards_embeddings, self.historical_cards), 13)

        hand_embed = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.hand_cards), [-1, 13, size_embed])
        hand_flat = conv1d("hand", hand_embed, 13)

        valid_embed = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.valid_cards), [-1, 13, size_embed])
        valid_flat = conv1d("valid", valid_embed, 13)

        expose_flat = conv1d("expose", tf.reshape(self.expose_info, [-1, 4, 1]), 4)
        void_flat = conv1d("void", self.void_info, 4)
        winning_flat = conv1d("winning", self.winning_info, 4)

        score_flat = conv2d_2("score", tf.nn.embedding_lookup(cards_embeddings, self.score_cards), size_score_card)

        concat_action_input = tf.concat([step, trick_nr, trick_flat,
                                         remaining_flat, must_flat, historical_flat, score_flat, valid_flat, hand_flat, expose_flat, void_flat, winning_flat], 
                                        axis=1, 
                                        name="concat_action_input")

        # 3. Policy Networks
        action_fd1 = tf.layers.dense(inputs=concat_action_input, units=2048, activation=activation_fn)
        action_fd2 = tf.layers.dense(inputs=action_fd1, units=512, activation=activation_fn)
        action_fd3 = tf.layers.dense(inputs=action_fd2, units=128, activation=activation_fn)
        action_fd = tf.layers.dense(inputs=action_fd3, units=64, activation=activation_fn)

        self.action_fc = tf.layers.dense(inputs=action_fd, units=13, activation=tf.nn.log_softmax)
        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.probs, self.action_fc), 1)))

        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = l2_penalty + self.policy_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

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


    def transpose(self, position, values):
        results = []
        for pos, value in zip(position, values):
            results.append([value[(pos[0]+idx)%4] for idx in range(4)])

        return results


    def policy_value(self, remaining_cards, \
                     trick_nr, trick_order, position, played_order, trick_cards, \
                     must_cards, historical_cards, score_cards, \
                     hand_cards, valid_cards, \
                     expose_info, void_info, winning_info):

        act_probs = self.session.run(self.action_fc,
                                     feed_dict={self.remaining_cards: remaining_cards,
                                                self.trick_nr: trick_nr,
                                                #self.trick_order: trick_order,
                                                #self.position: pos,
                                                self.played_order: played_order,
                                                self.trick_cards: trick_cards,
                                                self.must_cards: self.transpose(position, must_cards),
                                                self.historical_cards: self.transpose(position, historical_cards),
                                                self.score_cards: self.transpose(position, score_cards),
                                                self.hand_cards: hand_cards,
                                                self.valid_cards: valid_cards,
                                                self.expose_info: expose_info,
                                                self.void_info: self.transpose(position, void_info),
                                                self.winning_info: self.transpose(position, winning_info)})

        return np.exp(act_probs)


    def predict(self, trick_nr, state):
        remaining_cards, trick_nr, trick_order, pos, played_order, trick_cards, \
        must_cards, historical_cards, score_cards, hand_cards, valid_cards, \
        expose_info, void_info, winning_info = transform_game_info_to_nn(state, trick_nr)

        act_probs = self.policy_value([remaining_cards], \
                                      [[trick_nr]], [trick_order], [[pos]], [[played_order]], [trick_cards],\
                                      [must_cards], [historical_cards], [score_cards], [hand_cards], [valid_cards], \
                                      [expose_info], [void_info], [winning_info])

        return valid_cards, act_probs[0]


    def policy_value_fn(self, remaining_cards, \
                        trick_nr, trick_order, pos, played_order, trick_cards, \
                        must_cards, historical_cards, score_cards, hadn_cards, valid_cards, \
                        expose_info, void_info, winning_info):

        act_probs = self.policy_value(remaining_cards, \
                                      trick_nr, trick_order, pos, played_order, trick_cards,\
                                      must_cards, historical_cards, score_cards, hand_cards, valid_cards, \
                                      expose_info, void_info, winning_info)

        return act_probs


    def train_step(self, remaining_cards, \
                   trick_nr, trick_order, pos, played_order, trick_cards, \
                   must_cards, historical_cards, score_cards, hand_cards, valid_cards, \
                   expose_info, void_info, winning_info, probs, score, lr):

        loss, policy_loss, _ = self.session.run(
                [self.loss, self.policy_loss, self.optimizer],
                feed_dict={self.remaining_cards: remaining_cards,
                           self.trick_nr: trick_nr,
                           #self.trick_order: trick_order,
                           #self.position: pos,
                           self.played_order: played_order,
                           self.trick_cards: trick_cards,
                           self.must_cards: self.transpose(pos, must_cards),
                           self.historical_cards: self.transpose(pos, historical_cards),
                           self.score_cards: self.transpose(pos, score_cards),
                           self.hand_cards: hand_cards,
                           self.valid_cards: valid_cards,
                           self.expose_info: expose_info,
                           self.void_info: self.transpose(pos, void_info),
                           self.winning_info: self.transpose(pos, winning_info),
                           self.probs: probs,
                           self.score: self.transpose(pos, score),
                           self.learning_rate: lr})

        return loss, policy_loss


    def save_model(self, model_path):
        self.saver.save(self.session, model_path)
        print("save model in {}".format(model_path))


    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)


    def close(self):
        self.session.close()
