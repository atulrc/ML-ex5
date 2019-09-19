# Scientific and vector computation for python
import numpy as np

import utils
import linearRegCostFunction

def learnCurv(X, y, Xval, yval, lambda_=0):
    """
    Generates the train and cross validation set errors needed to plot a learning curve
    returns the train and cross validation set errors for a learning curve.

    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.

    Parameters
    ----------
    X : array_like
        The training dataset. Matrix with shape (m x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).

    Xval : array_like
        The validation dataset. Matrix with shape (m_val x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    yval : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).

    lambda_ : float, optional
        The regularization parameter.

    Returns
    -------
    error_train : array_like
        A vector of shape m. error_train[i] contains the training error for
        i examples.
    error_val : array_like
        A vecotr of shape m. error_val[i] contains the validation error for
        i training examples.

    Instructions
    ------------
    Fill in this function to return training errors in error_train and the
    cross validation errors in error_val. i.e., error_train[i] and
    error_val[i] should give you the errors obtained after training on i examples.

    Notes
    -----
    - You should evaluate the training error on the first i training
      examples (i.e., X[:i, :] and y[:i]).

      For the cross-validation error, you should instead evaluate on
      the _entire_ cross validation set (Xval and yval).

    - If you are using your cost function (linearRegCostFunction) to compute
      the training and cross validation error, you should call the function with
      the lambda argument set to 0. Do note that you will still need to use
      lambda when running the training to obtain the theta parameters.

    Hint
    ----
    You can loop over the examples with the following:

           for i in range(1, m+1):
               # Compute train/cross validation errors using training examples
               # X[:i, :] and y[:i], storing the result in
               # error_train[i-1] and error_val[i-1]
               ....
    """
    # Number of training examples
    m = y.size

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val   = np.zeros(m)

    # ====================== YOUR CODE HERE ======================

    for i in range(1, m + 1):
        theta_t = utils.trainLinearReg(linearRegCostFunction.linRegCostFun, X[:i], y[:i], lambda_ = lambda_)
        error_train[i - 1], _ = linearRegCostFunction.linRegCostFun(X[:i], y[:i], theta_t, lambda_ = 0)
        error_val[i - 1], _ = linearRegCostFunction.linRegCostFun(Xval, yval, theta_t, lambda_ = 0)

    # =============================================================
    return error_train, error_val
