# Scientific and vector computation for python
import numpy as np
import utils
import linearRegCostFunction

def valCurv(X, y, Xval, yval):
    """
    Generate the train and validation errors needed to plot a validation
    curve that we can use to select lambda_.

    Parameters
    ----------
    X : array_like
        The training dataset. Matrix with shape (m x n) where m is the
        total number of training examples, and n is the number of features
        including any polynomial features.

    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).

    Xval : array_like
        The validation dataset. Matrix with shape (m_val x n) where m is the
        total number of validation examples, and n is the number of features
        including any polynomial features.

    yval : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).

    Returns
    -------
    lambda_vec : list
        The values of the regularization parameters which were used in
        cross validation.

    error_train : list
        The training error computed at each value for the regularization
        parameter.

    error_val : list
        The validation error computed at each value for the regularization
        parameter.

    Instructions
    ------------
    Fill in this function to return training errors in `error_train` and
    the validation errors in `error_val`. The vector `lambda_vec` contains
    the different lambda parameters to use for each calculation of the
    errors, i.e, `error_train[i]`, and `error_val[i]` should give you the
    errors obtained after training with `lambda_ = lambda_vec[i]`.

    Note
    ----
    You can loop over lambda_vec with the following:

          for i in range(len(lambda_vec))
              lambda = lambda_vec[i]
              # Compute train / val errors when training linear
              # regression with regularization parameter lambda_
              # You should store the result in error_train[i]
              # and error_val[i]
              ....
    """
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    # ====================== YOUR CODE HERE ======================

    for i in range(len(lambda_vec)):
        lambda_try = lambda_vec[i]
        theta_t = utils.trainLinearReg(linearRegCostFunction.linRegCostFun, X, y, lambda_ = lambda_try)
        error_train[i], _ = linearRegCostFunction.linRegCostFun(X, y, theta_t, lambda_ = 0)
        error_val[i], _ = linearRegCostFunction.linRegCostFun(Xval, yval, theta_t, lambda_ = 0)

    # ============================================================
    return lambda_vec, error_train, error_val
