# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class PolicyValueNet(object):
    def __init__(self, model_file=None):
        # 1. input:
        self.states = tf.placeholder(tf.float32, shape=[None, 9], name="game_status")
        self.cards = tf.placeholder(tf.int32, shape=[None, 12], name="valid_cards")
        self.probs = tf.placeholder(tf.float32, shape=[None, 12], name="mcts_probs")
        self.scores = tf.placeholder(tf.float32, shape=[None, 1], name="scores")


        # Define the tensorflow neural network
        cards_embeddings = tf.Variable(tf.random_uniform([53, 16], -1.0, 1.0))
        self.cards_embed = tf.nn.embedding_lookup(cards_embeddings, self.cards)
        cards_flat = tf.reshape(self.cards_embed, [-1, 1])


        # 2. Common Networks Layers
        input = tf.reshape(tf.multiply(cards_flat, self.states), [-1, 12*16*9])
        input1 = tf.layers.dense(input, units=128, activation=tf.nn.relu)
        input2 = tf.layers.dense(input1, units=64, activation=tf.nn.relu)

        # 3. Policy Networks
        self.action_fc = tf.layers.dense(inputs=input2, units=12, activation=tf.nn.softmax)

        # 4 Value Networks
        self.evaluation_fc1 = tf.layers.dense(inputs=input2, units=64, activation=tf.nn.relu)
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1, units=1, activation=tf.nn.tanh)

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.scores, self.evaluation_fc2)
        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.probs, self.action_fc), 1)))

        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + l2_penalty + self.policy_loss

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        #self.entropy = tf.negative(tf.reduce_mean(
        #        tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)


    def policy_value(self, states, cards):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """

        log_act_probs, value = self.session.run([self.action_fc, self.evaluation_fc2],
                                                feed_dict={self.states: states,
                                                           self.cards: cards})

        act_probs = np.exp(log_act_probs)
        #value = self.session.run(self.evaluation_fc2,
        #                         feed_dict={self.states: states,
        #                                    self.cards: cards})

        return act_probs, value

    def policy_value_fn(self, states, cards):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """

        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))

        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])

        return act_probs, value


    def train_step(self, states, cards, probs, scores, lr):
        """perform a training step"""
        scores = np.reshape(scores, (-1, 1))

        loss, policy_loss, value_loss = self.session.run(
                [self.loss, self.policy_loss, self.value_loss, self.optimizer],
                feed_dict={self.states: states,
                           self.cards: cards,
                           self.probs: probs,
                           self.scores: scores,
                           self.learning_rate: lr})

        return loss, policy_loss, value_loss


    def save_model(self, model_path):
        self.saver.save(self.session, model_path)


    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
