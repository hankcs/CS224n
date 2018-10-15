#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    h = sigmoid(np.dot(data, W1) + b1)  # M * H
    yhat = softmax(np.dot(h, W2) + b2)  # M * Dy
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    grad_cost1 = (yhat - labels) / data.shape[0]  # M * Dy, 最后一层，平均化
    tmp = np.dot(grad_cost1, W2.T)  # M * H
    grad_cost2 = sigmoid_grad(h) * tmp / data.shape[0]  # M * H， 隐含层，全部加和后的误差再平均化，标准梯度下降

    cost = np.sum(-np.log(yhat[labels == 1])) / data.shape[0]

    gradW2 = np.dot(h.T, grad_cost1)  # H * Dy
    gradb2 = np.sum(grad_cost1, 0)  # 1 * Dy
    gradW1 = np.dot(data.T, grad_cost2)  # Dx * H
    gradb1 = np.sum(grad_cost2, 0)  # 1 * H

    ### END YOUR CODE

    ### STANDRED CODE
    cost = np.sum(-np.log(yhat[labels==1])) / data.shape[0]

    d3 = (yhat - labels) / data.shape[0]
    gradW2 = np.dot(h.T, d3)
    gradb2 = np.sum(d3,0,keepdims=True)

    dh = np.dot(d3,W2.T)
    grad_h = sigmoid_grad(h) * dh

    gradW1 = np.dot(data.T,grad_h)
    gradb1 = np.sum(grad_h,0)
    ### END STANDRED CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
