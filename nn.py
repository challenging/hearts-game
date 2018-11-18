import os

import numpy as np
import tensorflow as tf

from nn_utils import transform_game_info_to_nn


class PolicyValueNet(object):
    def __init__(self, model_file=None, size_embed=32):
        # 1. input:
        self.remaining_cards = tf.placeholder(tf.int32, shape=[None, 52], name="remaining_cards")
        self.trick_cards = tf.placeholder(tf.int32, shape=[None, 3], name="trick_cards")
        self.must_cards_1 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_1")
        self.must_cards_2 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_2")
        self.must_cards_3 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_3")
        self.must_cards_4 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_4")
        self.score_cards_1 = tf.placeholder(tf.int32, shape=[None, 15], name="score_cards_1")
        self.score_cards_2 = tf.placeholder(tf.int32, shape=[None, 15], name="score_cards_2")
        self.score_cards_3 = tf.placeholder(tf.int32, shape=[None, 15], name="score_cards_3")
        self.score_cards_4 = tf.placeholder(tf.int32, shape=[None, 15], name="score_cards_4")
        self.hand_cards = tf.placeholder(tf.int32, shape=[None, 13], name="hand_cards")
        self.valid_cards = tf.placeholder(tf.int32, shape=[None, 13], name="valid_cards")
        self.expose_info = tf.placeholder(tf.float32, shape=[None, 4], name="expose_info")
        self.probs = tf.placeholder(tf.float32, shape=[None, 13], name="probs")
        self.score = tf.placeholder(tf.float32, shape=[None, 4], name="score")

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # Define the embedding layer
        cards_embeddings = tf.Variable(tf.random_uniform([53, size_embed], -1.0, 1.0))

        self.remaining_cards_embed = tf.nn.embedding_lookup(cards_embeddings, self.remaining_cards)
        remaining_cards_flat = tf.reshape(self.remaining_cards_embed, [-1, 52*size_embed])

        self.trick_cards_embed = tf.nn.embedding_lookup(cards_embeddings, self.trick_cards)
        trick_cards_flat = tf.reshape(self.trick_cards_embed, [-1, 3*size_embed])

        self.must_cards_embed_1 = tf.nn.embedding_lookup(cards_embeddings, self.must_cards_1)
        must_cards_flat_1 = tf.reshape(self.must_cards_embed_1, [-1, 4*size_embed])

        self.must_cards_embed_2 = tf.nn.embedding_lookup(cards_embeddings, self.must_cards_2)
        must_cards_flat_2 = tf.reshape(self.must_cards_embed_2, [-1, 4*size_embed])

        self.must_cards_embed_3 = tf.nn.embedding_lookup(cards_embeddings, self.must_cards_3)
        must_cards_flat_3 = tf.reshape(self.must_cards_embed_3, [-1, 4*size_embed])

        self.must_cards_embed_4 = tf.nn.embedding_lookup(cards_embeddings, self.must_cards_4)
        must_cards_flat_4 = tf.reshape(self.must_cards_embed_4, [-1, 4*size_embed])

        self.score_cards_embed_1 = tf.nn.embedding_lookup(cards_embeddings, self.score_cards_1)
        score_cards_flat_1 = tf.reshape(self.score_cards_embed_1, [-1, 15*size_embed])

        self.score_cards_embed_2 = tf.nn.embedding_lookup(cards_embeddings, self.score_cards_2)
        score_cards_flat_2 = tf.reshape(self.score_cards_embed_2, [-1, 15*size_embed])

        self.score_cards_embed_3 = tf.nn.embedding_lookup(cards_embeddings, self.score_cards_3)
        score_cards_flat_3 = tf.reshape(self.score_cards_embed_3, [-1, 15*size_embed])

        self.score_cards_embed_4 = tf.nn.embedding_lookup(cards_embeddings, self.score_cards_4)
        score_cards_flat_4 = tf.reshape(self.score_cards_embed_4, [-1, 15*size_embed])

        self.hand_cards_embed = tf.nn.embedding_lookup(cards_embeddings, self.hand_cards)
        hand_cards_flat = tf.reshape(self.hand_cards_embed, [-1, 13*size_embed])

        self.valid_cards_embed = tf.nn.embedding_lookup(cards_embeddings, self.valid_cards)
        valid_cards_flat = tf.reshape(self.valid_cards_embed, [-1, 13*size_embed])

        concat_action_input = tf.concat([remaining_cards_flat, \
                                         trick_cards_flat, \
                                         must_cards_flat_1, \
                                         must_cards_flat_2, \
                                         must_cards_flat_3, \
                                         must_cards_flat_4, \
                                         hand_cards_flat, \
                                         valid_cards_flat], axis=1, name="concat_action_input")


        # 2. Common Networks Layers
        input1 = tf.layers.dense(concat_action_input, units=1024, activation=tf.nn.relu)
        input2 = tf.layers.dense(input1, units=512, activation=tf.nn.relu)
        input3 = tf.layers.dense(input2, units=128, activation=tf.nn.relu)
        input4 = tf.layers.dense(input3, units=32, activation=tf.nn.relu)

        # 3. Policy Networks
        self.action_fc = tf.layers.dense(inputs=input4, units=13, activation=tf.nn.log_softmax)
        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.probs, self.action_fc), 1)))*64

        # 4 Value Networks
        input5 = tf.concat([input4,
                            score_cards_flat_1, \
                            score_cards_flat_2, \
                            score_cards_flat_3, \
                            score_cards_flat_4,
                            self.expose_info], axis=1, name="concat_value_input")

        input6 = tf.layers.dense(input5, units=1024, activation=tf.nn.relu)
        input7 = tf.layers.dense(input6, units=512, activation=tf.nn.relu)
        input8 = tf.layers.dense(input7, units=128, activation=tf.nn.relu)
        input9 = tf.layers.dense(input8, units=32, activation=tf.nn.relu)

        self.evaluation_fc1 = tf.layers.dense(inputs=input9, units=16, activation=tf.nn.relu)
        self.evaluation_fc = tf.layers.dense(inputs=self.evaluation_fc1, units=4, activation=tf.nn.relu)
        self.value_loss = tf.losses.mean_squared_error(self.score, self.evaluation_fc)

        # 4-2. Policy Loss function
        #self.policy_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(self.probs, self.action_fc)), 1))

        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + l2_penalty + self.policy_loss

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


    def policy_value(self, remaining_cards, trick_cards, \
                     must_cards_1, must_cards_2, must_cards_3, must_cards_4, \
                     score_cards_1, score_cards_2, score_cards_3, score_cards_4, \
                     hand_cards, valid_cards, expose_info):

        act_probs, value = self.session.run([self.action_fc, self.evaluation_fc],
                                            feed_dict={self.remaining_cards: remaining_cards,
                                                       self.trick_cards: trick_cards,
                                                       self.must_cards_1: must_cards_1,
                                                       self.must_cards_2: must_cards_2,
                                                       self.must_cards_3: must_cards_3,
                                                       self.must_cards_4: must_cards_4,
                                                       self.score_cards_1: score_cards_1,
                                                       self.score_cards_2: score_cards_2,
                                                       self.score_cards_3: score_cards_3,
                                                       self.score_cards_4: score_cards_4,
                                                       self.hand_cards: hand_cards,
                                                       self.valid_cards: valid_cards,
                                                       self.expose_info: expose_info})

        return np.exp(act_probs), value


    def predict(self, trick_nr, state):
        remaining_cards, trick_cards, must_cards_1, must_cards_2, must_cards_3, must_cards_4, \
            score_cards_1, score_cards_2, score_card_3, score_cards_4, hand_cards, valid_cards, expose_info = transform_game_info_to_nn(state, trick_nr)

        act_probs, act_values = self.policy_value([remaining_cards], [trick_cards],\
                                                  [must_cards_1], [must_cards_2], [must_cards_3], [must_cards_4],\
                                                  [score_cards_1], [score_cards_2], [score_card_3], [score_cards_4],\
                                                  [hand_cards], [valid_cards], [expose_info])

        return valid_cards, act_probs[0], act_values[0]


    def policy_value_fn(self, remaining_cards, trick_cards, \
                        must_cards_1, must_cards_2, must_cards_3, must_cards_4, \
                        score_cards_1, score_cards_2, score_cards_3, score_cards_4, \
                        hadn_cards, valid_cards, expose_info):

        act_probs, act_values = self.policy_value(remaining_cards, trick_cards,\
                                                  must_cards_1, must_cards_2, must_cards_3, must_cards_4,\
                                                  score_cards_1, score_cards_2, score_card_3, score_cards_4,\
                                                  hand_cards, valid_cards, expose_info)

        return act_probs, act_values


    def train_step(self, remaining_cards, trick_cards, \
                   must_cards_1, must_cards_2, must_cards_3, must_cards_4, \
                   score_cards_1, score_cards_2, score_cards_3, score_cards_4, \
                   hand_cards, valid_cards, expose_info, probs, score, lr):

        loss, policy_loss, value_loss, _ = self.session.run(
                [self.loss, self.policy_loss, self.value_loss, self.optimizer],
                feed_dict={self.remaining_cards: remaining_cards,
                           self.trick_cards: trick_cards,
                           self.must_cards_1: must_cards_1,
                           self.must_cards_2: must_cards_2,
                           self.must_cards_3: must_cards_3,
                           self.must_cards_4: must_cards_4,
                           self.score_cards_1: score_cards_1,
                           self.score_cards_2: score_cards_2,
                           self.score_cards_3: score_cards_3,
                           self.score_cards_4: score_cards_4,
                           self.hand_cards: hand_cards,
                           self.valid_cards: valid_cards,
                           self.expose_info: expose_info,
                           self.probs: probs,
                           self.score: score,
                           self.learning_rate: lr})

        return loss, policy_loss, value_loss


    def save_model(self, model_path):
        self.saver.save(self.session, model_path)


    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)


    def close(self):
        self.session.close()
