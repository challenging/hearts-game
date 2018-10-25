# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class PolicyValueNet(object):
    def __init__(self, model_file=None):
        # 1. input:
        self.remaining_cards = tf.placeholder(tf.int32, shape=[None, 52], name="remaining_cards")
        self.trick_cards = tf.placeholder(tf.int32, shape=[None, 3], name="trick_cards")
        self.must_cards_1 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_1")
        self.must_cards_2 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_2")
        self.must_cards_3 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_3")
        self.must_cards_4 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_4")
        self.score_cards_1 = tf.placeholder(tf.int32, shape=[None, 52], name="score_cards_1")
        self.score_cards_2 = tf.placeholder(tf.int32, shape=[None, 52], name="score_cards_2")
        self.score_cards_3 = tf.placeholder(tf.int32, shape=[None, 52], name="score_cards_3")
        self.score_cards_4 = tf.placeholder(tf.int32, shape=[None, 52], name="score_cards_4")
        self.probs = tf.placeholder(tf.float32, shape=[None, 52], name="probs")
        self.score = tf.placeholder(tf.float32, shape=[None, 4], name="score")

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # Define the embedding layer
        cards_embeddings = tf.Variable(tf.random_uniform([53, 16], -1.0, 1.0))

        self.remaining_cards_embed = tf.nn.embedding_lookup(cards_embeddings, self.remaining_cards)
        remaining_cards_flat = tf.reshape(self.remaining_cards_embed, [-1, 52*16])

        self.trick_cards_embed = tf.nn.embedding_lookup(cards_embeddings, self.trick_cards)
        trick_cards_flat = tf.reshape(self.trick_cards_embed, [-1, 3*16])

        self.must_cards_embed_1 = tf.nn.embedding_lookup(cards_embeddings, self.must_cards_1)
        must_cards_flat_1 = tf.reshape(self.must_cards_embed_1, [-1, 4*16])

        self.must_cards_embed_2 = tf.nn.embedding_lookup(cards_embeddings, self.must_cards_2)
        must_cards_flat_2 = tf.reshape(self.must_cards_embed_2, [-1, 4*16])

        self.must_cards_embed_3 = tf.nn.embedding_lookup(cards_embeddings, self.must_cards_3)
        must_cards_flat_3 = tf.reshape(self.must_cards_embed_3, [-1, 4*16])

        self.must_cards_embed_4 = tf.nn.embedding_lookup(cards_embeddings, self.must_cards_4)
        must_cards_flat_4 = tf.reshape(self.must_cards_embed_4, [-1, 4*16])

        self.score_cards_embed_1 = tf.nn.embedding_lookup(cards_embeddings, self.score_cards_1)
        score_cards_flat_1 = tf.reshape(self.score_cards_embed_1, [-1, 52*16])

        self.score_cards_embed_2 = tf.nn.embedding_lookup(cards_embeddings, self.score_cards_2)
        score_cards_flat_2 = tf.reshape(self.score_cards_embed_2, [-1, 52*16])

        self.score_cards_embed_3 = tf.nn.embedding_lookup(cards_embeddings, self.score_cards_3)
        score_cards_flat_3 = tf.reshape(self.score_cards_embed_3, [-1, 52*16])

        self.score_cards_embed_4 = tf.nn.embedding_lookup(cards_embeddings, self.score_cards_4)
        score_cards_flat_4 = tf.reshape(self.score_cards_embed_4, [-1, 52*16])

        concat_input = tf.concat([remaining_cards_flat, \
                                  trick_cards_flat, \
                                  must_cards_flat_1, \
                                  must_cards_flat_2, \
                                  must_cards_flat_3, \
                                  must_cards_flat_4, \
                                  score_cards_flat_1, \
                                  score_cards_flat_2, \
                                  score_cards_flat_3, \
                                  score_cards_flat_4], axis=1, name="concat_input")
        #concat_input = tf.concat([remaining_cards_flat, trick_cards_flat], axis=1, name="concat_input")


        # 2. Common Networks Layers
        input1 = tf.layers.dense(concat_input, units=1024, activation=tf.nn.relu)
        input2 = tf.layers.dense(input1, units=256, activation=tf.nn.relu)
        input3 = tf.layers.dense(input2, units=64, activation=tf.nn.relu)

        # 3. Policy Networks
        self.action_fc = tf.layers.dense(inputs=input3, units=52, activation=tf.nn.softmax)

        # 4 Value Networks
        self.evaluation_fc1 = tf.layers.dense(inputs=input3, units=4, activation=tf.nn.relu)
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1, units=4, activation=tf.nn.tanh)

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.score, self.evaluation_fc2)
        #self.value_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.score, self.evaluation_fc2), 1)))
        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.probs, self.action_fc), 1)))

        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + l2_penalty + self.policy_loss

        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)


    def policy_value(self, remaining_cards, trick_cards, \
                     must_cards_1, must_cards_2, must_cards_3, must_cards_4, \
                     score_cards_1, score_cards_2, score_cards_3, score_cards_4):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """

        log_act_probs, value = self.session.run([self.action_fc, self.evaluation_fc2],
                                                feed_dict={self.remaining_cards: remaining_cards,
                                                           self.trick_cards: trick_cards,
                                                           self.must_cards_1: must_cards_1,
                                                           self.must_cards_2: must_cards_2,
                                                           self.must_cards_3: must_cards_3,
                                                           self.must_cards_4: must_cards_4,
                                                           self.score_cards_1: score_cards_1,
                                                           self.score_cards_2: score_cards_2,
                                                           self.score_cards_3: score_cards_3,
                                                           self.score_cards_4: score_cards_4})

        act_probs = np.exp(log_act_probs)

        return act_probs, value

    def policy_value_fn(self, remaining_cards, trick_cards, \
                        must_cards_1, must_cards_2, must_cards_3, must_cards_4,\
                        score_cards_1, score_cards_2, score_card_3, score_cards_4):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """

        act_probs, value = self.policy_value(remaining_cards, trick_cards,\
                                             must_cards_1, must_cards_2, must_cards_3, must_cards_4,\
                                             score_cards_1, score_cards_2, score_card_3, score_cards_4)

        return act_probs, value


    def train_step(self, remaining_cards, trick_cards, \
                   must_cards_1, must_cards_2, must_cards_3, must_cards_4, \
                   score_cards_1, score_cards_2, score_cards_3, score_cards_4, \
                   probs, score, lr):

        #print(" probs:", probs)
        #print(" score:", score)
        #print("    lr:", lr)

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
                           self.probs: probs,
                           self.score: score,
                           self.learning_rate: lr})

        return loss, policy_loss, value_loss


    def save_model(self, model_path):
        self.saver.save(self.session, model_path)


    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
