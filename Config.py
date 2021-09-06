import numpy as np
import tensorflow as tf

class Config():
    def __init__(self, batch_size, input_shape, seed, chunk_size, rnn_size, n_chunks):
        self.BATCH_SIZE = batch_size
        self.INPUT_SHAPE = input_shape
        self.SEED = seed

        self.CHUNK_SIZE = chunk_size
        self.RNN_SIZE = rnn_size
        self.N_CHUNKS = n_chunks

        self.train_data_node = tf.placeholder(tf.float32,
                                              shape=(self.BATCH_SIZE, self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 1), name='train_data_node')
        self.train_labels_node = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, 2),name='train_data_label')

        self.test_data_node = tf.placeholder(tf.float32,
                                              shape=(1, self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 1),name='test_data_node')
        self.test_labels_node = tf.placeholder(tf.float32, shape=(1, 2), name='test_data_label')

        self.conv1_weights = tf.Variable(tf.truncated_normal([2, 8, 1, 96], stddev=0.1, seed=self.SEED))
        self.conv1_biases = tf.Variable(tf.zeros([96]))

        self.conv2_weights = tf.Variable(tf.truncated_normal([2, 8, 96, 96], stddev=0.1, seed=self.SEED))
        self.conv2_biases = tf.Variable(tf.zeros([96]))

        self.conv3_weights = tf.Variable(tf.truncated_normal([2, 8, 96, 96], stddev=0.1, seed=self.SEED))
        self.conv3_biases = tf.Variable(tf.constant(0.2, shape=[96]))

        self.conv4_weights = tf.Variable(tf.truncated_normal([2, 8, 96, 96], stddev=0.1, seed=self.SEED))
        self.conv4_biases = tf.Variable(tf.zeros([96]))

        with tf.variable_scope('gru1') as scope:
            self.gru1 = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([tf.contrib.rnn.core_rnn_cell.GRUCell(96)])
        with tf.variable_scope('gru2') as scope:
            self.gru2 = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([tf.contrib.rnn.core_rnn_cell.GRUCell(96)])

        self.fc_weights = tf.Variable(tf.truncated_normal([96, 2], stddev=0.1, seed=self.SEED))
        self.fc_biases = tf.Variable(tf.constant(0.1, shape=[2]))