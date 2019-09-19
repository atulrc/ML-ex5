# Scientific and vector computation for python
import numpy as np

def polyFeat(X, p):
    """
    Maps X (1D vector) into the p-th power.

    Parameters
    ----------
    X : array_like
        A data vector of size m, where m is the number of examples.

    p : int
        The polynomial power to map the features.

    Returns
    -------
    X_poly : array_like
        A matrix of shape (m x p) where p is the polynomial
        power and m is the number of examples. That is:

        X_poly[i, :] = [X[i], X[i]**2, X[i]**3 ...  X[i]**p]

    Instructions
    ------------
    Given a vector X, return a matrix X_poly where the p-th column of
    X contains the values of X to the p-th power.
    """
    # You need to return the following variables correctly.
    X_poly = np.zeros((X.shape[0], p))

    # ====================== YOUR CODE HERE ======================

    for i in range(p):
        X_poly[:, i] = X[:, 0] ** (i + 1)

    # ============================================================
    return X_poly
