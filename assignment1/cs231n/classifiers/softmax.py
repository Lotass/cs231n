import numpy as np
from random import shuffle
from past.builtins import xrange


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

  for i in range(num_train):
    row_score = X[i].dot(W)
    row_score -= np.max(row_score)

    row_score_exp_sum = np.sum(np.exp(row_score))
    correct_label_score = np.exp(row_score[y[i]])
    loss += -np.log(correct_label_score / row_score_exp_sum)

    for c in range(num_classes):
      dW[:,c] += ( (np.exp(row_score[c]) / row_score_exp_sum) - (c == y[i])  ) * X[i]
  
  dW /= num_train
  dW += reg * W
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

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

  scoreMatrix = X.dot(W);
  scoreMatrix -= np.max(scoreMatrix,axis=1,keepdims=True)

  scoreSumExp = np.sum(np.exp(scoreMatrix),axis=1,keepdims=True) # sum of exp over each training example
  probDist = np.exp(scoreMatrix) / scoreSumExp

  loss = np.sum(-np.log(probDist[np.arange(num_train),y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  probDist[np.arange(num_train),y] -= 1 
  
  dW = X.T.dot(probDist);
  dW /= num_train
  dW += reg * W

  return loss, dW
