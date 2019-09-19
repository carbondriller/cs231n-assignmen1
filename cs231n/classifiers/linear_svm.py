import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # training example X[i] maps to category y[i], according to data provided
    # let's see how much score has category y[i] obtained? (from our model W above)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # the category j has been (incorrectly) given a score higher than the score of actual category y[i]
        # adding this incorrect margin to loss will adjust this
        loss += margin
        
        # Source: https://github.com/lightaime/cs231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py#L37,
        # https://math.stackexchange.com/a/2572319/195667
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  
  scores = X.dot(W)
  # make y a column vector
  correct_class_score = np.reshape(y, (y.shape[0], 1))
  # broadcasdt 1 to correct_class_score vector, then broadcast that to scores
  margins = scores - (correct_class_score + 1) # margins is (N,C)
  
  # only keep where margin > 0.
  negative_idxs = margins < 0
  margins[ negative_idxs ] = 0 # https://stackoverflow.com/a/28431064/3578289
  
  # Select margins (2D array) elements at index equal to y (1D array) elements
  # https://gist.github.com/carbondriller/9462c87649a775ca77d94fe5768b89df
  row_idxs = np.arange(N)
  # set each row (i-th training exmple)'s actual label y[i] to 0, so it doesn't get included in sum
  margins[ row_idxs, y ] = 0
  
  loss = np.sum(margins)
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
