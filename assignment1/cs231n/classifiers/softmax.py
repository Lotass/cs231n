import numpy as np
from random import shuffle


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
    num_train = X.shape[0]
    num_classes = W.shape[1]
    num_dimensions = W.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)

        sum_scores = np.exp(scores).sum()
        probs = np.zeros_like(scores)

        for k in range(num_classes):
            probs[k] = np.exp(scores[k]) / sum_scores

        dscores = probs.copy()
        dscores[y[i]] -= 1

        dW += X[i].T.reshape(num_dimensions,
                             1).dot(dscores.reshape(1, num_classes))
        loss += -np.log(probs[y[i]])

    loss += reg * np.square(W).sum()
    loss /= num_train

    dW += 2 * reg * W
    dW /= num_train

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

    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    ## trick to avoid numerical ins
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    loss = np.sum(-np.log(probs[range(num_train), y]))
    loss += reg * np.square(W).sum()
    loss /= num_train

    dprobs = probs.copy()
    dprobs[range(num_train),y] -= 1

    dW = X.T.dot(dprobs)
    dW += reg * 2 * W
    dW /= num_train
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
