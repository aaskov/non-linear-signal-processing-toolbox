# -*- coding: utf-8 -*-
import numpy as np


def kernel(X, Y):
    """Euclidian distance between X and Y.

    Args:
        X (ndarray): Input X.\n
        Y (ndarray): Input Y.

    Yields:
        K (ndarray): Euclideian distance matrix between X and Y.

    Example:
        Calculate the distance matrix to itself.

        >>> K = kernel(X, X)
    """
    dist_1_1 = sum(X*X, 0)
    dist_2_2 = sum(Y*Y, 0)

    A = np.array(np.asmatrix(dist_1_1).T.repeat(len(Y[0]), 1))
    B = np.array(np.asmatrix(dist_2_2).repeat(len(X[0]), 0))
    C = X.T.dot(Y)

    K = A + B - 2*C
    return(K)


def kernel_svm_rbf(X, Y, sigma):
    """Get the elementwise RBF kernel transformation.

    Args:
        X (ndarray): Input X.\n
        Y (ndarray): Input Y. \n
        sigma (float): Std. diviation of the Guassian expression.

    Yields:
        K_transformed (ndarray): Transformed kernel.

    """
    K = kernel(X, Y)
    K_transformed = np.exp(-K/(2*sigma**2))
    return(K_transformed)
