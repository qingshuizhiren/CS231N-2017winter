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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    escores = np.exp(scores)
    correct_class_escore = escores[y[i]]
    prob = correct_class_escore/np.sum(escores)
    loss -= np.log(prob)
    for j in xrange(num_classes):
       dW[:,j] += escores[j] / np.sum(escores) * (X[i].T)
       if j == y[i]:
         dW[:,y[i]] -= X[i].T
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss / num_train 
  loss += 0.5 * reg * np.sum(W * W)
  dW = dW / num_train
  dW += reg * W
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
  pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  max_scores = np.max(scores, axis=1).reshape(-1,1)
  scores = scores - max_scores
  escores = np.exp(scores)
  correct_class_escore = escores[xrange(num_train), list(y)].reshape(-1, 1)
  sum_escores = np.sum(escores, axis=1).reshape(-1,1)
  loss = -np.sum(np.log(correct_class_escore/sum_escores)) / num_train
  loss += 0.5 * reg * np.sum(W * W)
  prob = escores / sum_escores
  prob[xrange(num_train), y] -= 1
  dW = (X.T).dot(prob) / num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

