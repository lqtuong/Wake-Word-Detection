import numpy as np
import tensorflow as tf
from Config import Config

class Model():
    def __init__(self):
        self.config = Config(batch_size=50, input_shape=(128,22), seed=42, chunk_size=28, rnn_size=192, n_chunks=28)
        self.conv = None
        self.relu = None
        self.pool = None
        self.stack = None
        self.input_rnn = None
        self.gru1_out = None
        self.gru2_out = None
        self.dropout = None
        self.flat = None
        self.p_y_X = None
        self.step = 0
        self.test_predict = tf.nn.softmax(self.build(data=self.config.test_data_node))

    def build(self, data, train=False):

        with tf.variable_scope('CRNN') as scope:
            self.conv = tf.nn.conv2d(data, self.config.conv1_weights, strides=[1, 2, 2, 1], padding='SAME')
            self.relu = tf.nn.relu(tf.nn.bias_add(self.conv, self.config.conv1_biases))
            self.pool = tf.nn.max_pool(self.relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            self.conv = tf.nn.conv2d(self.pool, self.config.conv2_weights, strides=[1, 2, 2, 1], padding='SAME')
            self.relu = tf.nn.relu(tf.nn.bias_add(self.conv, self.config.conv2_biases))
            self.pool = tf.nn.max_pool(self.relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            self.conv = tf.nn.conv2d(self.pool, self.config.conv3_weights, strides=[1, 2, 2, 1], padding='SAME')
            self.relu = tf.nn.relu(tf.nn.bias_add(self.conv, self.config.conv3_biases))
            self.pool = tf.nn.max_pool(self.relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            self.conv = tf.nn.conv2d(self.pool, self.config.conv4_weights, strides=[1, 2, 2, 1], padding='SAME')
            self.relu = tf.nn.relu(tf.nn.bias_add(self.conv, self.config.conv4_biases))
            self.pool = tf.nn.max_pool(self.relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            self.stack = tf.unstack(self.pool)
            self.input_rnn = tf.transpose(tf.concat(self.stack, 0), [0, 2, 1])

        with tf.variable_scope('rnn_gru') as scope:
            if self.step > 0:
                tf.get_variable_scope().reuse_variables()
            self.step += 1

            self.gru1_out, state = tf.nn.dynamic_rnn(self.config.gru1, self.input_rnn, dtype=tf.float32, scope='gru1')
            self.gru2_out, state = tf.nn.dynamic_rnn(self.config.gru2, self.gru1_out, dtype=tf.float32, scope='gru2')

            self.gru2_out = tf.transpose(self.gru2_out, [1, 0, 2])
            self.gru2_out = tf.gather(self.gru2_out, int(self.gru2_out.get_shape()[0]) - 1)
            self.dropout = tf.nn.dropout(self.gru2_out, 0.3)

            self.flat = tf.reshape(self.dropout, [-1, self.config.fc_weights.get_shape().as_list()[0]])
            self.p_y_X = tf.add(tf.matmul(self.flat, self.config.fc_weights), self.config.fc_biases)

            return self.p_y_X


