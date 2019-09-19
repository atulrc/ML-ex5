# Scientific and vector computation for python
import numpy as np

def linRegCostFun(X, y, theta, lambda_=0.0):
    """
    Compute cost and gradient for regularized linear regression
    with multiple variables. Computes the cost of using theta as
    the parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The dataset. Matrix with shape (m x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y : array_like
        The functions values at each datapoint. A vector of
        shape (m, ).

    theta : array_like
        The parameters for linear regression. A vector of shape (n+1,).

    lambda_ : float, optional
        The regularization parameter.

    Returns
    -------
    J : float
        The computed cost function.

    grad : array_like
        The value of the cost function gradient w.r.t theta.
        A vector of shape (n+1, ).

    Instructions
    ------------
    Compute the cost and gradient of regularized linear regression for
    a particular choice of theta.
    You should set J to the cost and grad to the gradient.
    """
    # Initialize some useful values
    m = y.size # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================

    h = X.dot(theta)
    J = (1 / (2 * m)) * np.sum(np.square(h - y)) + (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))

    grad = (1 / m) * (h - y).dot(X)

    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]

    # ============================================================
    return J, grad
