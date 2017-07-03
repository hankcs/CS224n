#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1: A window into NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import time
import logging
from datetime import datetime

import tensorflow as tf

from util import print_sentence, write_conll
from data_util import load_and_preprocess_data, load_embeddings, read_conll, ModelHelper
from ner_model import NERModel
from defs import LBLS

# from report import Report

logger = logging.getLogger("hw3.q1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    TODO: Fill in what n_window_features should be, using n_word_features and window_size.
    """
    n_word_features = 2  # Number of features for every word in the input.
    window_size = 1  # The size of the window to use.
    ### YOUR CODE HERE
    n_window_features = (2 * window_size + 1) * n_word_features  # The total number of features used for each window.
    ### END YOUR CODE
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001

    def __init__(self, output_path=None):
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/window/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "window_predictions.conll"


def make_windowed_data(data, start, end, window_size=1):
    """Uses the input sequences in @data to construct new windowed data points.

    TODO: In the code below, construct a window from each word in the
    input sentence by concatenating the words @window_size to the left
    and @window_size to the right to the word. Finally, add this new
    window data point and its label. to windowed_data.

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels.
        start: the featurized `start' token to be used for windows at the very
            beginning of the sentence.
        end: the featurized `end' token to be used for windows at the very
            end of the sentence.
        window_size: the length of the window to construct.
    Returns:
        a new list of data points, corresponding to each window in the
        sentence. Each data point consists of a list of
        @n_window_features features (corresponding to words from the
        window) to be used in the sentence and its NER label.
        If start=[5,8] and end=[6,8], the above example should return
        the list
        [([5, 8, 1, 9, 2, 9], 1),
         ([1, 9, 2, 9, 3, 8], 1),
         ...
         ]
    """

    windowed_data = []
    for sentence, labels in data:
    ### YOUR CODE HERE (5-20 lines)
        orig_n = len(sentence)
        # extend sentence
        sentence = [start] * window_size + sentence + [end] * window_size
        l = 0  # index labels
        # loop over the original sentence
        for i in range(window_size, orig_n + window_size):
            temp_feats = []
            # loop over the window for feature in original sentence
            for j in range(i - window_size, i + window_size + 1):
                temp_feats.extend(sentence[j])

            # put token features together with label:
            temp_f_l = (temp_feats, labels[l])
            # put into windowed data:
            windowed_data.append(temp_f_l)
            # iterate our labels index
            l += 1
    ### END YOUR CODE
    return windowed_data


class WindowModel(NERModel):
    """
    Implements a feedforward neural network with an embedding layer and
    single hidden layer.
    This network will predict what label (e.g. PER) should be given to a
    given token (e.g. Manning) by  using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_window_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None,), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~3-5 lines)
        self.input_placeholder = tf.placeholder(tf.int32, [None, self.config.n_window_features])
        self.labels_placeholder = tf.placeholder(tf.int32, [None,])
        self.dropout_placeholder = tf.placeholder(tf.float32)
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE (~5-10 lines)
        feed_dict = {self.input_placeholder: inputs_batch, \
                     self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_window_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_window_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.
        Returns:
            embeddings: tf.Tensor of shape (None, n_window_features*embed_size)
        """
        ### YOUR CODE HERE (!3-5 lines)
        embedded = tf.Variable(self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(embedded,self.input_placeholder)
        embeddings = tf.reshape(embeddings, [-1, self.config.n_window_features * self.config.embed_size])
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Recall that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        When creating a new variable, use the tf.get_variable function
        because it lets us specify an initializer.

        Use tf.contrib.layers.xavier_initializer to initialize matrices.
        This is TensorFlow's implementation of the Xavier initialization
        trick we used in last assignment.

        Note: tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of dropout_rate.

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        ### YOUR CODE HERE (~10-20 lines)
        b1 = tf.get_variable(name='b1', shape = [self.config.hidden_size,], \
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b2 = tf.get_variable(name='b2', shape = [self.config.n_classes], \
                             initializer=tf.contrib.layers.xavier_initializer(seed=2))

        W = tf.get_variable(name='W', shape = [self.config.n_window_features * self.config.embed_size, self.config.hidden_size], \
                            initializer=tf.contrib.layers.xavier_initializer(seed=3))

        U = tf.get_variable(name='U', shape = [self.config.hidden_size, self.config.n_classes], \
                            initializer=tf.contrib.layers.xavier_initializer(seed=4))

        z1 = tf.matmul(x,W) + b1
        h = tf.nn.relu(z1)
        h_drop = tf.nn.dropout(h, dropout_rate)
        pred = tf.matmul(h_drop,U) + b2
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Remember that you can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
        implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-5 lines)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder)
        loss = tf.reduce_mean(loss)
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        adam_optim = tf.train.AdamOptimizer(self.config.lr)
        train_op = adam_optim.minimize(loss)
        ### END YOUR CODE
        return train_op

    def preprocess_sequence_data(self, examples):
        return make_windowed_data(examples, start=self.helper.START, end=self.helper.END,
                                  window_size=self.config.window_size)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        # pdb.set_trace()
        i = 0
        for sentence, labels in examples_raw:
            labels_ = preds[i:i + len(sentence)]
            i += len(sentence)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(WindowModel, self).__init__(helper, config, report)
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()


def test_make_windowed_data():
    sentences = [[[1, 1], [2, 0], [3, 3]]]
    sentence_labels = [[1, 2, 3]]
    data = zip(sentences, sentence_labels)
    w_data = make_windowed_data(data, start=[5, 0], end=[6, 0], window_size=1)

    assert len(w_data) == sum(len(sentence) for sentence in sentences)

    assert w_data == [
        ([5, 0] + [1, 1] + [2, 0], 1,),
        ([1, 1] + [2, 0] + [3, 3], 2,),
        ([2, 0] + [3, 3] + [6, 0], 3,),
    ]


def do_test1(_):
    logger.info("Testing make_windowed_data")
    test_make_windowed_data()
    logger.info("Passed!")


def do_test2(args):
    logger.info("Testing implementation of WindowModel")
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = WindowModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)

    logger.info("Model did not crash!")
    logger.info("Passed!")


def do_train(args):
    # Set up some parameters.
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None  # Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = WindowModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                # Save predictions in a text file.
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                predictions = [[LBLS[l] for l in preds] for preds in predictions]
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_conll(f, output)
                with open(model.config.eval_output, 'w') as f:
                    for sentence, labels, predictions in output:
                        print_sentence(f, sentence, labels, predictions)


def do_evaluate(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    input_data = read_conll(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = WindowModel(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, input_data):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)


def do_shell(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = WindowModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)

            print("""Welcome!
You can use this shell to explore the behavior of your model.
Please enter sentences with spaces between tokens, e.g.,
input> Germany 's representative to the European Union 's veterinary committee .
""")
            while True:
                # Create simple REPL
                try:
                    sentence = raw_input("input> ")
                    tokens = sentence.strip().split(" ")
                    for sentence, _, predictions in model.output(session, [(tokens, ["O"] * len(tokens))]):
                        predictions = [LBLS[l] for l in predictions]
                        print_sentence(sys.stdout, sentence, [""] * len(tokens), predictions)
                except EOFError:
                    print("Closing session.")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test1', help='')
    command_parser.set_defaults(func=do_test1)

    command_parser = subparsers.add_parser('test2', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/tiny.conll",
                                help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/tiny.conll",
                                help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt",
                                help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt",
                                help="Path to word vectors file")
    command_parser.set_defaults(func=do_test2)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/train.conll",
                                help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/dev.conll",
                                help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt",
                                help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt",
                                help="Path to word vectors file")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="data/dev.conll",
                                help="Training data")
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt",
                                help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt",
                                help="Path to word vectors file")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('shell', help='')
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt",
                                help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt",
                                help="Path to word vectors file")
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
