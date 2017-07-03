#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2(c): Recurrent neural nets for NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("hw3.q2.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class RNNCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our RNN cell implementation that allows us to play
    nicely with TensorFlow.
    """

    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the RNN equations are:

        h_t = sigmoid(x_t W_x + h_{t-1} W_h + b)

        TODO: In the code below, implement an RNN cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_x, W_h, b to be variables of the apporiate shape
              using the `tf.get_variable' functions. Make sure you use
              the names "W_x", "W_h" and "b"!
            - Compute @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~6-10 lines)
            W_x = tf.get_variable('W_x', shape=(self.input_size, self._state_size), dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            W_h = tf.get_variable('W_h', shape=(self._state_size, self._state_size), dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', shape=(self._state_size), dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

            new_state = tf.nn.sigmoid(tf.matmul(state, W_h) + tf.matmul(inputs, W_x) + b)
            ### END YOUR CODE ###
        # For an RNN , the output and state are the same (N.B. this
        # isn't true for an LSTM, though we aren't using one of those in
        # our assignment)
        output = new_state
        return output, new_state


def test_rnn_cell():
    with tf.Graph().as_default():
        with tf.variable_scope("test_rnn_cell"):
            x_placeholder = tf.placeholder(tf.float32, shape=(None, 3))
            h_placeholder = tf.placeholder(tf.float32, shape=(None, 2))

            with tf.variable_scope("rnn"):
                tf.get_variable("W_x", initializer=np.array(np.eye(3, 2), dtype=np.float32))
                tf.get_variable("W_h", initializer=np.array(np.eye(2, 2), dtype=np.float32))
                tf.get_variable("b", initializer=np.array(np.ones(2), dtype=np.float32))

            tf.get_variable_scope().reuse_variables()
            cell = RNNCell(3, 2)
            y_var, ht_var = cell(x_placeholder, h_placeholder, scope="rnn")

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                x = np.array([
                    [0.4, 0.5, 0.6],
                    [0.3, -0.2, -0.1]], dtype=np.float32)
                h = np.array([
                    [0.2, 0.5],
                    [-0.3, -0.3]], dtype=np.float32)
                y = np.array([
                    [0.832, 0.881],
                    [0.731, 0.622]], dtype=np.float32)
                ht = y

                y_, ht_ = session.run([y_var, ht_var], feed_dict={x_placeholder: x, h_placeholder: h})
                print("y_ = " + str(y_))
                print("ht_ = " + str(ht_))

                assert np.allclose(y_, ht_), "output and state should be equal."
                assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."


def do_test(_):
    logger.info("Testing rnn_cell")
    test_rnn_cell()
    logger.info("Passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the RNN cell implemented as part of Q2 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
