from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Num classes = num cols of w
    C = W.shape[1]
    # Number of examples
    N = X.shape[0]
    
    for i in range(N):
        scores = np.dot(X[i,:], W)
        true_score = scores[y[i]]
        
        # Calculate normalize probabilities
        log_c = -np.max(scores)
        sum_sj = np.sum(np.exp(scores + log_c))
        
        # Example specific loss and add to total
        s_yi = true_score + log_c
        L_i = -np.log(np.exp(s_yi)/sum_sj)
        loss = loss + L_i
        
        # Calculate gradients for each class
        for c in range(C):
            # Pull score against all possible classes, apply activation
            # and calculate derivative of softmax as dW
            class_score = scores[c]
            class_softmax = np.exp(class_score + log_c)/sum_sj
            dW[:,c] = X[i,:]*(class_softmax-1*(c==y[i]))
            
    # normalize and regularize
    dW = dW / N
    loss = loss / N
    
    loss = loss + (0.5*reg*np.sum(W*W))
    dW = dW + reg*W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
