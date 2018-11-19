import os

import tensorflow as tf

from nn import PolicyValueNet


class CNNPolicyValueNet(PolicyValueNet):
    def __init__(self, model_file=None, size_embed=32):
        size_score_card = 15

        # 1. input:
        self.remaining_cards = tf.placeholder(tf.int32, shape=[None, 52], name="remaining_cards")
        self.trick_cards = tf.placeholder(tf.int32, shape=[None, 3], name="trick_cards")
        self.must_cards_1 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_1")
        self.must_cards_2 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_2")
        self.must_cards_3 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_3")
        self.must_cards_4 = tf.placeholder(tf.int32, shape=[None, 4], name="must_cards_4")
        self.score_cards_1 = tf.placeholder(tf.int32, shape=[None, size_score_card], name="score_cards_1")
        self.score_cards_2 = tf.placeholder(tf.int32, shape=[None, size_score_card], name="score_cards_2")
        self.score_cards_3 = tf.placeholder(tf.int32, shape=[None, size_score_card], name="score_cards_3")
        self.score_cards_4 = tf.placeholder(tf.int32, shape=[None, size_score_card], name="score_cards_4")
        self.hand_cards = tf.placeholder(tf.int32, shape=[None, 13], name="hand_cards")
        self.valid_cards = tf.placeholder(tf.int32, shape=[None, 13], name="valid_cards")
        self.expose_info = tf.placeholder(tf.float32, shape=[None, 4], name="expose_info")
        self.probs = tf.placeholder(tf.float32, shape=[None, 13], name="probs")
        self.score = tf.placeholder(tf.float32, shape=[None, 4], name="score")

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # Define the embedding layer
        cards_embeddings = tf.Variable(tf.random_uniform([53, size_embed], -1.0, 1.0))

        remaining_embed = tf.nn.embedding_lookup(cards_embeddings, self.remaining_cards)
        remaining_cards_embed = tf.reshape(remaining_embed, [-1, 52, size_embed, 1], name="reshape_remaining_embed")
        remaining_conv1 = tf.layers.conv2d(inputs=remaining_cards_embed,
                                           filters=8, 
                                           kernel_size=[3, 3],
                                           padding="same", 
                                           data_format="channels_last",
                                           activation=tf.nn.relu)

        remaining_conv2 = tf.layers.conv2d(inputs=remaining_conv1,
                                           filters=16, 
                                           kernel_size=[3, 3],
                                           padding="same", 
                                           data_format="channels_last",
                                           activation=tf.nn.relu)

        remaining_conv3 = tf.layers.conv2d(inputs=remaining_conv2,
                                           filters=32, 
                                           kernel_size=[3, 3],
                                           padding="same", 
                                           data_format="channels_last",
                                           activation=tf.nn.relu)

        remaining_conv = tf.layers.conv2d(inputs=remaining_conv3,
                                          filters=4, 
                                          kernel_size=[1, 1],
                                          padding="same", 
                                          data_format="channels_last",
                                          activation=tf.nn.relu)

        remaining_flat = tf.reshape(remaining_conv, [-1, 4*52*size_embed], name="reshape_remaining")

        """
        remaining_fd1 = tf.layers.dense(inputs=remaining_flat,
                                        units=1024,
                                        activation=tf.nn.relu)
        remaining_fd2 = tf.layers.dense(inputs=remaining_fd1,
                                        units=256,
                                        activation=tf.nn.relu)
        remaining_fd = tf.layers.dense(inputs=remaining_fd2,
                                        units=64,
                                        activation=tf.nn.relu)
        """


        must_embed_1 = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.must_cards_1), [-1, 4, size_embed, 1], name="reshape_must1")
        must_embed_2 = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.must_cards_2), [-1, 4, size_embed, 1], name="reshape_must2")
        must_embed_3 = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.must_cards_3), [-1, 4, size_embed, 1], name="reshape_must3")
        must_embed_4 = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.must_cards_4), [-1, 4, size_embed, 1], name="reshape_must4")
        concat_must = tf.concat([must_embed_1, must_embed_2, must_embed_3, must_embed_4], axis=3, name="concat_must")

        def conv2d(name, input, size):
            conv1 = tf.layers.conv2d(inputs=input,
                                     filters=8,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     data_format="channels_last",
                                     activation=tf.nn.relu)

            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=16,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     data_format="channels_last",
                                     activation=tf.nn.relu)

            conv = tf.layers.conv2d(inputs=conv2,
                                    filters=4,
                                    kernel_size=[1, 1],
                                    padding="same",
                                    data_format="channels_last",
                                    activation=tf.nn.relu)

            return tf.reshape(conv, [-1, 4*size*size_embed], name="reshape_{}".format(name))

        must_flat = conv2d("must", concat_must, 4)
        #must_fd = tf.layers.dense(inputs=must_flat, units=64, activation=tf.nn.relu)

        trick_embed = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.trick_cards), [-1, 3, size_embed, 1])
        trick_flat = conv2d("trick", trick_embed, 3*1)
        #trick_fd = tf.layers.dense(inputs=trick_flat, units=64, activation=tf.nn.relu)

        hand_embed = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.hand_cards), [-1, 13, size_embed, 1])
        hand_flat = conv2d("hand", hand_embed, 13*1)
        #hand_fd1 = tf.layers.dense(inputs=hand_flat, units=256, activation=tf.nn.relu)
        #hand_fd = tf.layers.dense(inputs=hand_fd1, units=64, activation=tf.nn.relu)

        valid_embed = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.valid_cards), [-1, 13, size_embed, 1])
        valid_flat = conv2d("valid", valid_embed, 13*1)
        #valid_fd1 = tf.layers.dense(inputs=valid_flat, units=256, activation=tf.nn.relu)
        #valid_fd = tf.layers.dense(inputs=valid_fd1, units=64, activation=tf.nn.relu)

        score_embed_1 = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.score_cards_1), [-1, size_score_card, size_embed, 1])
        score_embed_2 = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.score_cards_2), [-1, size_score_card, size_embed, 1])
        score_embed_3 = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.score_cards_3), [-1, size_score_card, size_embed, 1])
        score_embed_4 = tf.reshape(tf.nn.embedding_lookup(cards_embeddings, self.score_cards_4), [-1, size_score_card, size_embed, 1])
        concat_score = tf.concat([score_embed_1, score_embed_2, score_embed_3, score_embed_4], axis=3, name="concat_score")

        score_conv1 = tf.layers.conv2d(inputs=concat_score,
                                       filters=8,
                                       kernel_size=[3, 3],
                                       padding="same",
                                       data_format="channels_last",
                                       activation=tf.nn.relu)

        score_conv2 = tf.layers.conv2d(inputs=score_conv1,
                                       filters=16,
                                       kernel_size=[3, 3],
                                       padding="same",
                                       data_format="channels_last",
                                       activation=tf.nn.relu)

        score_conv3 = tf.layers.conv2d(inputs=score_conv2,
                                       filters=32,
                                       kernel_size=[3, 3],
                                       padding="same",
                                       data_format="channels_last",
                                       activation=tf.nn.relu)

        score_conv = tf.layers.conv2d(inputs=score_conv3,
                                      filters=4,
                                      kernel_size=[1, 1],
                                      padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)

        score_flat = tf.reshape(score_conv, [-1, 4*size_score_card*size_embed], name="reshape_score")
        #score_fd1 = tf.layers.dense(score_flat, units=1024, activation=tf.nn.relu)
        #score_fd2 = tf.layers.dense(score_fd1, units=256, activation=tf.nn.relu)
        #score_fd = tf.layers.dense(score_fd2, units=64, activation=tf.nn.relu)

        concat_action_input = tf.concat([remaining_flat, 
                                         must_flat, 
                                         trick_flat, 
                                         hand_flat, 
                                         valid_flat], 
                                        axis=1, 
                                        name="concat_action_input")

        input1 = tf.layers.dense(concat_action_input, units=4096, activation=tf.nn.relu)
        input2 = tf.layers.dense(input1, units=1024, activation=tf.nn.relu)
        input3 = tf.layers.dense(input2, units=256, activation=tf.nn.relu)
        input4 = tf.layers.dense(input3, units=64, activation=tf.nn.relu)

        # 3. Policy Networks
        self.action_fc = tf.layers.dense(inputs=input4, units=13, activation=tf.nn.log_softmax)
        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.probs, self.action_fc), 1)))

        concat_value_input = tf.concat([input2,
                                        score_flat,
                                        self.expose_info],
                                        axis=1,
                                        name="concat_value_input")

        evaluation_fc1 = tf.layers.dense(inputs=concat_value_input, units=2048, activation=tf.nn.relu)
        evaluation_fc2 = tf.layers.dense(inputs=evaluation_fc1, units=256, activation=tf.nn.relu)
        evaluation_fc3 = tf.layers.dense(inputs=evaluation_fc2, units=64, activation=tf.nn.relu)
        evaluation_fc4 = tf.layers.dense(inputs=evaluation_fc3, units=16, activation=tf.nn.relu)
        self.evaluation_fc = tf.layers.dense(inputs=evaluation_fc4, units=4, activation=tf.nn.relu)
        self.value_loss = tf.losses.absolute_difference(self.score, self.evaluation_fc)

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
