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


def kernel_gaussian(K, sig2):
    """Get the elementwise Gaussian kernel transformation.
    
    Args:
        K (ndarray): Kernel matrix ('see kernel(X,Y)').\n
        sig2 (float): Variance of the Gaussian expression.
    
    Yields:
        K_transformed (ndarray): Transformed kernel.
    """
    K_transformed = np.exp(-K/(2*sig2))
    return(K_transformed)


def kernel_gp(K, sig2, beta):
    """Get the elementwise kernel transformation.
    Note: We assume the noice has mean = 0, variance = beta,  there is no
    correlation between the samples in terms of noise. If one or more of
    these assumptions does not hold, we could have been including that to
    the model.
    
    Args:
        K (ndarray): Kernel matrix ('see kernel(X,Y)').\n
        sig2 (float): Variance of the Gaussian expression.\n
        beta (float): Noise term added on distances to themselfs.
    
    Yields:
        K_transformed (ndarray): Transformed kernel.
    """
    S = np.shape(K)
    K_transformed = np.exp(-K/(2*sig2)) + (1/beta)*np.eye(S[0])
    return(K_transformed)