# -*- coding: utf-8 -*-
import numpy as np

def kernel(X, Y):
    """Euclidian distance between X and Y."""
    dist_1_1 = sum(X*X, 0)
    dist_2_2 = sum(Y*Y, 0)

    A = np.array(np.asmatrix(dist_1_1).T.repeat(len(Y[0]), 1))
    B = np.array(np.asmatrix(dist_2_2).repeat(len(X[0]), 0))
    C = X.T.dot(Y)

    K = A + B - 2*C
    return(K)


def svm_kernel_rbf(X, Y, sigma):
    """SVM: Radial basis function kernel"""
    K = kernel(X,Y)
    K_transformed = np.exp(-K/(2*sigma**2))    
    return(K_transformed)
