# -*- coding: utf-8 -*-
import numpy as np
from svm_kernels import kernel_svm_rbf


def svm_classify(newdata, SVM):
    """SVM classifier.

    Args:
        newdata (ndarray): Input test set (of unseen data).\n
        SVM (dictinary): The trained SVM from 'svm_train'.

    Yields:
        prediction (ndarray): The list of predictions .

    Example:
        Calculate the distance matrix to itself.

        >>> K = kernel(X, X)
    """
    # Unpack the SVM dictionary
    sv = SVM['support_vector']
    alpha_hat = SVM['alpha_hat']
    scale = SVM['scale']
    bias = SVM['bias']
    shift = SVM['shift']
    sigma = SVM['sigma']

    # Shift and scale the support vectors
    for c in range(len(sv[0])):
        sv[:, c] = scale[c] * (sv[:, c] + shift[c])

    # Shift and scale the data
    for c in range(len(newdata[0])):
        newdata[:, c] = scale[c] * (newdata[:, c] + shift[c])

    # Get the RBF kernel
    G = np.asmatrix(kernel_svm_rbf(sv.T, newdata.T, sigma)).T

    # Classify new data
    f = np.dot(G, alpha_hat.T) + bias

    # Class prediction is determined by the sign of the dot product above
    prediction = np.sign(f)
    prediction = np.array(prediction)

    return(prediction)
