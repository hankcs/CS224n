#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2016-17: Homework 3
util.py: General utility routines
Arun Chaganty <chaganty@stanford.edu>
"""

from __future__ import division

import sys
import time
import logging
import StringIO
from collections import defaultdict, Counter, OrderedDict
import numpy as np
from numpy import array, zeros, allclose

logger = logging.getLogger("hw3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def read_conll(fstream):
    """
    Reads a input stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @returns a list of examples [(tokens), (labels)]. @tokens and @labels are lists of string.
    """
    ret = []

    current_toks, current_lbls = [], []
    for line in fstream:
        line = line.strip()
        if len(line) == 0 or line.startswith("-DOCSTART-"):
            if len(current_toks) > 0:
                assert len(current_toks) == len(current_lbls)
                ret.append((current_toks, current_lbls))
            current_toks, current_lbls = [], []
        else:
            assert "\t" in line, r"Invalid CONLL format; expected a '\t' in {}".format(line)
            tok, lbl = line.split("\t")
            current_toks.append(tok)
            current_lbls.append(lbl)
    if len(current_toks) > 0:
        assert len(current_toks) == len(current_lbls)
        ret.append((current_toks, current_lbls))
    return ret

def test_read_conll():
    input_ = [
        "EU	ORG",
        "rejects	O",
        "German	MISC",
        "call	O",
        "to	O",
        "boycott	O",
        "British	MISC",
        "lamb	O",
        ".	O",
        "",
        "Peter	PER",
        "Blackburn	PER",
        "",
        ]
    output = [
        ("EU rejects German call to boycott British lamb .".split(), "ORG O MISC O O O MISC O O".split()),
        ("Peter Blackburn".split(), "PER PER".split())
        ]

    assert read_conll(input_) == output

def write_conll(fstream, data):
    """
    Writes to an output stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @data a list of examples [(tokens), (labels), (predictions)]. @tokens, @labels, @predictions are lists of string.
    """
    for cols in data:
        for row in zip(*cols):
            fstream.write("\t".join(row))
            fstream.write("\n")
        fstream.write("\n")

def test_write_conll():
    input = [
        ("EU rejects German call to boycott British lamb .".split(), "ORG O MISC O O O MISC O O".split()),
        ("Peter Blackburn".split(), "PER PER".split())
        ]
    output = """EU	ORG
rejects	O
German	MISC
call	O
to	O
boycott	O
British	MISC
lamb	O
.	O

Peter	PER
Blackburn	PER

"""
    output_ = StringIO.StringIO()
    write_conll(output_, input)
    output_ = output_.getvalue()
    assert output == output_

def load_word_vector_mapping(vocab_fstream, vector_fstream):
    """
    Load word vector mapping using @vocab_fstream, @vector_fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.
    """
    ret = OrderedDict()
    for vocab, vector in zip(vocab_fstream, vector_fstream):
        vocab = vocab.strip()
        vector = vector.strip()
        ret[vocab] = array(list(map(float, vector.split())))

    return ret

def test_load_word_vector_mapping():
    vocab = """UUUNKKK
the
,
.
of
and
in""".split("\n")
    vector = """0.172414 -0.091063 0.255125 -0.837163 0.434872 -0.499848 -0.042904 -0.059642 -0.635087 -0.458795 -0.105671 0.506513 -0.105105 -0.405678 0.493365 0.408807 0.401635 -0.817805 0.626340 0.580636 -0.246996 -0.008515 -0.671140 0.301865 -0.439651 0.247694 -0.291402 0.873009 0.216212 0.145576 -0.211101 -0.352360 0.227651 -0.118416 0.371816 0.261296 0.017548 0.596692 -0.485722 -0.369530 -0.048807 0.017960 -0.040483 0.111193 0.398039 0.162765 0.408946 0.005343 -0.107523 -0.079821
-0.454847 1.002773 -1.406829 -0.016482 0.459856 -0.224457 0.093396 -0.826833 -0.530674 1.211044 -0.165133 0.174454 -1.130952 -0.612020 -0.024578 -0.168508 0.320113 0.774229 -0.360418 1.483124 -0.230922 0.301055 -0.119924 0.601642 0.694616 -0.304431 -0.414284 0.667385 0.171208 -0.334842 -0.459286 -0.534202 0.533660 -0.379468 -0.378721 -0.240499 -0.446272 0.686113 0.662359 -0.865312 0.861331 -0.627698 -0.569544 -1.228366 -0.152052 1.589123 0.081337 0.182695 -0.593022 0.438300
-0.408797 -0.109333 -0.099279 -0.857098 -0.150319 -0.456398 -0.781524 -0.059621 0.302548 0.202162 -0.319892 -0.502241 -0.014925 0.020889 1.506245 0.247530 0.385598 -0.170776 0.325960 0.267304 0.157673 0.125540 -0.971452 -0.485595 0.487857 0.284369 -0.062811 -1.334082 0.744133 0.572701 1.009871 -0.457229 0.938059 0.654805 -0.430244 -0.697683 -0.220146 0.346002 -0.388637 -0.149513 0.011248 0.818728 0.042615 -0.594237 -0.646138 0.568898 0.700328 0.290316 0.293722 0.828779
-0.583585 0.413481 -0.708189 0.168942 0.238435 0.789011 -0.566401 0.177570 -0.244441 0.328214 -0.319583 -0.468558 0.520323 0.072727 1.792047 -0.781348 -0.636644 0.070102 -0.247090 0.110990 0.182112 1.609935 -1.081378 0.922773 -0.605783 0.793724 0.476911 -1.279422 0.904010 -0.519837 1.235220 -0.149456 0.138923 0.686835 -0.733707 -0.335434 -1.865440 -0.476014 -0.140478 -0.148011 0.555169 1.356662 0.850737 -0.484898 0.341224 -0.056477 0.024663 1.141509 0.742001 0.478773
-0.811262 -1.017245 0.311680 -0.437684 0.338728 1.034527 -0.415528 -0.646984 -0.121626 0.589435 -0.977225 0.099942 -1.296171 0.022671 0.946574 0.204963 0.297055 -0.394868 0.028115 -0.021189 -0.448692 0.421286 0.156809 -0.332004 0.177866 0.074233 0.299713 0.148349 1.104055 -0.172720 0.292706 0.727035 0.847151 0.024006 -0.826570 -1.038778 -0.568059 -0.460914 -1.290872 -0.294531 0.663751 -0.646503 0.499024 -0.804777 -0.402926 -0.292201 0.348031 0.215414 0.043492 0.165281
-0.156019 0.405009 -0.370058 -1.417499 0.120639 -0.191854 -0.251213 -0.883898 -0.025010 0.150738 1.038723 0.038419 0.036411 -0.289871 0.588898 0.618994 0.087019 -0.275657 -0.105293 -0.536067 -0.181410 0.058034 0.552306 -0.389803 -0.384800 -0.470717 0.800593 -0.166609 0.702104 0.876092 0.353401 -0.314156 0.618290 0.804017 -0.925911 -1.002050 -0.231087 0.590011 -0.636952 -0.474758 0.169423 1.293482 0.609088 -0.956202 -0.013831 0.399147 0.436669 0.116759 -0.501962 1.308268
-0.008573 -0.731185 -1.108792 -0.358545 0.507277 -0.050167 0.751870 0.217678 -0.646852 -0.947062 -1.187739 0.490993 -1.500471 0.463113 1.370237 0.218072 0.213489 -0.362163 -0.758691 -0.670870 0.218470 1.641174 0.293220 0.254524 0.085781 0.464454 0.196361 -0.693989 -0.384305 -0.171888 0.045602 1.476064 0.478454 0.726961 -0.642484 -0.266562 -0.846778 0.125562 -0.787331 -0.438503 0.954193 -0.859042 -0.180915 -0.944969 -0.447460 0.036127 0.654763 0.439739 -0.038052 0.991638""".split("\n")

    wvs = load_word_vector_mapping(vocab, vector)
    assert "UUUNKKK" in wvs
    assert allclose(wvs["UUUNKKK"], array([0.172414, -0.091063, 0.255125, -0.837163, 0.434872, -0.499848, -0.042904, -0.059642, -0.635087, -0.458795, -0.105671, 0.506513, -0.105105, -0.405678, 0.493365, 0.408807, 0.401635, -0.817805, 0.626340, 0.580636, -0.246996, -0.008515, -0.671140, 0.301865, -0.439651, 0.247694, -0.291402, 0.873009, 0.216212, 0.145576, -0.211101, -0.352360, 0.227651, -0.118416, 0.371816, 0.261296, 0.017548, 0.596692, -0.485722, -0.369530, -0.048807, 0.017960, -0.040483, 0.111193, 0.398039, 0.162765, 0.408946, 0.005343, -0.107523, -0.079821]))
    assert "the" in wvs
    assert "of" in wvs
    assert "and" in wvs

def window_iterator(seq, n=1, beg="<s>", end="</s>"):
    """
    Iterates through seq by returning windows of length 2n+1
    """
    for i in range(len(seq)):
        l = max(0, i-n)
        r = min(len(seq), i+n+1)
        ret = seq[l:r]
        if i < n:
            ret = [beg,] * (n-i) + ret
        if i+n+1 > len(seq):
            ret = ret + [end,] * (i+n+1 - len(seq))
        yield ret

def test_window_iterator():
    assert list(window_iterator(list("abcd"), n=0)) == [["a",], ["b",], ["c",], ["d"]]
    assert list(window_iterator(list("abcd"), n=1)) == [["<s>","a","b"], ["a","b","c",], ["b","c","d",], ["c", "d", "</s>",]]

def one_hot(n, y):
    """
    Create a one-hot @n-dimensional vector with a 1 in position @i
    """
    if isinstance(y, int):
        ret = zeros(n)
        ret[y] = 1.0
        return ret
    elif isinstance(y, list):
        ret = zeros((len(y), n))
        ret[np.arange(len(y)),y] = 1.0
        return ret
    else:
        raise ValueError("Expected an int or list got: " + y)


def to_table(data, row_labels, column_labels, precision=2, digits=4):
    """Pretty print tables.
    Assumes @data is a 2D array and uses @row_labels and @column_labels
    to display table.
    """
    # Convert data to strings
    line = "%0"+str(digits)+"."+str(precision)+"f"
    data = [[line%v for v in row] for row in data]
    cell_width = max(
        max(map(len, row_labels)),
        max(map(len, column_labels)),
        max(max(map(len, row)) for row in data))
    def c(s):
        """adjust cell output"""
        return s + " " * (cell_width - len(s))
    ret = ""
    ret += "\t".join(map(c, column_labels)) + "\n"
    for l, row in zip(row_labels, data):
        ret += "\t".join(map(c, [l] + row)) + "\n"
    return ret

class ConfusionMatrix(object):
    """
    A confusion matrix stores counts of (true, guessed) labels, used to
    compute several evaluation metrics like accuracy, precision, recall
    and F1.
    """

    def __init__(self, labels, default_label=None):
        self.labels = labels
        self.default_label = default_label if default_label is not None else len(labels) -1
        self.counts = defaultdict(Counter)

    def update(self, gold, guess):
        """Update counts"""
        self.counts[gold][guess] += 1

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self.counts[l][l_] for l_,_ in enumerate(self.labels)] for l,_ in enumerate(self.labels)]
        return to_table(data, self.labels, ["go\\gu"] + self.labels, precision=0, digits=0)

    def summary(self, quiet=False):
        """Summarize counts"""
        keys = range(len(self.labels))
        data = []
        macro = array([0., 0., 0., 0.])
        micro = array([0., 0., 0., 0.])
        default = array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__] for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0

            # update micro/macro averages
            micro += array([tp, fp, tn, fn])
            macro += array([acc, prec, rec, f1])
            if l != self.default_label: # Count count for everything that is not the default label!
                default += array([tp, fp, tn, fn])

            data.append([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / len(keys))

        # default average
        tp, fp, tn, fn = default
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])

        # Macro and micro average.
        return to_table(data, self.labels + ["micro","macro","not-O"], ["label", "acc", "prec", "rec", "f1"])

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)

def print_sentence(output, sentence, labels, predictions):

    spacings = [max(len(sentence[i]), len(labels[i]), len(predictions[i])) for i in range(len(sentence))]
    # Compute the word spacing
    output.write("x : ")
    for token, spacing in zip(sentence, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y*: ")
    for token, spacing in zip(labels, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y': ")
    for token, spacing in zip(predictions, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")
