#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3(d): Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
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
        Remember the GRU equations are:

        z_t = sigmoid(x_t U_z + h_{t-1} W_z + b_z)
        r_t = sigmoid(x_t U_r + h_{t-1} W_r + b_r)
        o_t = tanh(x_t U_o + r_t * h_{t-1} W_o + b_o)
        h_t = z_t * h_{t-1} + (1 - z_t) * o_t

        TODO: In the code below, implement an GRU cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_r, U_r, b_r, W_z, U_z, b_z and W_o, U_o, b_o to
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
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
            ### YOUR CODE HERE (~20-30 lines)
            initFunc = tf.contrib.layers.xavier_initializer(uniform=True) # set to False for gradient clipping
            W_r = tf.get_variable('W_r', [self.state_size, self.state_size], initializer=initFunc, dtype = tf.float32)
            U_r = tf.get_variable('U_r', [self.input_size, self.state_size], initializer=initFunc, dtype = tf.float32)
            b_r = tf.get_variable('b_r', [self.state_size,], initializer=tf.constant_initializer(0), dtype = tf.float32)
            W_z = tf.get_variable('W_z', [self.state_size, self.state_size], initializer=initFunc, dtype = tf.float32)
            U_z = tf.get_variable('U_z', [self.input_size, self.state_size], initializer=initFunc, dtype = tf.float32)
            b_z = tf.get_variable('b_z', [self.state_size,], initializer=tf.constant_initializer(0), dtype = tf.float32)    ## Recommend on Piazza
            W_o = tf.get_variable('W_o', [self.state_size, self.state_size], initializer=initFunc, dtype = tf.float32)
            U_o = tf.get_variable('U_o', [self.input_size, self.state_size], initializer=initFunc, dtype = tf.float32)
            b_o = tf.get_variable('b_o', [self.state_size,], initializer=tf.constant_initializer(0), dtype = tf.float32)

            z_t = tf.sigmoid(tf.matmul(inputs, U_z) + tf.matmul(state, W_z) + b_z)
            r_t = tf.sigmoid(tf.matmul(inputs, U_r) + tf.matmul(state, W_r) + b_r)
            o_t = tf.tanh(tf.matmul(inputs, U_o) + tf.matmul(r_t * state, W_o) + b_o)
            new_state = z_t * state + (1 - z_t) * o_t
            ### END YOUR CODE ###
        # For a GRU, the output and state are the same (N.B. this isn't true
        # for an LSTM, though we aren't using one of those in our
        # assignment)
        output = new_state
        return output, new_state

def test_gru_cell():
    with tf.Graph().as_default():
        with tf.variable_scope("test_gru_cell"):
            x_placeholder = tf.placeholder(tf.float32, shape=(None,3))
            h_placeholder = tf.placeholder(tf.float32, shape=(None,2))

            with tf.variable_scope("gru"):
                tf.get_variable("U_r", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("W_r", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_r",  initializer=np.array(np.ones(2), dtype=np.float32))
                tf.get_variable("U_z", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("W_z", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_z",  initializer=np.array(np.ones(2), dtype=np.float32))
                tf.get_variable("U_o", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("W_o", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_o",  initializer=np.array(np.ones(2), dtype=np.float32))

            tf.get_variable_scope().reuse_variables()
            cell = GRUCell(3, 2)
            y_var, ht_var = cell(x_placeholder, h_placeholder, scope="gru")

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
                    [ 0.320, 0.555],
                    [-0.006, 0.020]], dtype=np.float32)
                ht = y

                y_, ht_ = session.run([y_var, ht_var], feed_dict={x_placeholder: x, h_placeholder: h})
                print("y_ = " + str(y_))
                print("ht_ = " + str(ht_))

                assert np.allclose(y_, ht_), "output and state should be equal."
                assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."

def do_test(_):
    logger.info("Testing gru_cell")
    test_gru_cell()
    logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the GRU cell implemented as part of Q3 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
