import numpy as np
from random import shuffle
from past.builtins import xrange
from math import exp, log


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in xrange(num_train):
        # batch size = 1
        scores = X[i].dot(W)
        # D_score r.s.t. W . i.e. D_score.shape = (D,)
        D_score = X[i].T
        # nominator is A float
        nominator = exp(scores[y[i]])
        denominator = 0
        for j in range(num_classes):
            denominator += exp(scores[j])
        softmax = nominator / denominator
        # D_softmax r.s.t scores . i.e. D_softmax.shape = S.shape =(C,)
        D_softmax = (np.exp(scores) / denominator)
        D_softmax[y[i]] -= 1
        # D_softmax = D_softmax.reshape(1)
        loss += - log(softmax)
        # dW = D_softmax * D_score
        dW += np.matmul(D_score.reshape(-1, 1), D_softmax.reshape(1, -1))

    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = np.dot(X, W)
    exp_scores = np.exp(scores)
    softmax = exp_scores[np.arange(num_train), y] / np.sum(exp_scores, axis=1)
    Li = -np.log(softmax)
    loss = np.sum(Li) / num_train + reg * np.sum(W * W)

    D_scores = exp_scores / np.sum(exp_scores, axis=1).reshape(-1, 1)
    D_scores[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, D_scores)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
