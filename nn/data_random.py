# -*- coding: utf-8 -*-
"""
Neural network - nsp
"""
import numpy as np


def get_random_data(Ni, t_Nh, No, ptrain, ptest, noise):
    """Create input and output data from a 'teacher' network. The outputs are
    contaminated with additive white noise.

    Args:
        Ni: Number of external inputs to net.\n
        t_Nh: Number of hidden units for the 'teacher' net.\n
        No: Number of output units.\n
        ptrain: Number of training examples.\n
        ptest: Number of test examples.\n
        noise: Relatice amplitude of additive noise.

    Yields:
        tr_i, te_i: Input for training and test set.\n
        tr_t, te_t: Target values.

    Examples:
        Load data
        >>> tr_i, tr_t, te_i, te_t = getdata(...)
    """
    # Initialize 'teacher' weights
    TWi = np.random.randn(t_Nh, Ni+1)  # Input to hidden
    TWo = np.random.randn(No, t_Nh+1)  # Hidden to output

    # Create random inputs
    tr_i = np.random.randn(ptrain, Ni)
    te_i = np.random.randn(ptest, Ni)

    # Determine 'teacher' outputs
    tr_t = np.concatenate((np.tanh(np.concatenate((tr_i, np.ones((ptrain, 1))), 1).dot(TWi.T)), np.ones((ptrain, 1))), 1).dot(TWo.T)
    te_t = np.concatenate((np.tanh(np.concatenate((te_i, np.ones((ptest, 1))), 1).dot(TWi.T)), np.ones((ptest, 1))), 1).dot(TWo.T)

    # Add noise to each output unit column
    if tr_t.shape[1] > 1:
        amp = [np.std(arr) for arr in tr_t]

        for u in np.arange(No):
            tr_t[:, u] = tr_t[:, u] + noise*amp[u]*np.random.randn(ptrain, 1)
            te_t[:, u] = te_t[:, u] + noise*amp[u]*np.random.randn(ptest, 1)
    else:
        amp = np.std(tr_t)
        tr_t = tr_t + noise*amp*np.random.randn(ptrain, 1)
        te_t = te_t + noise*amp*np.random.randn(ptest, 1)

    return (tr_i, tr_t, te_i, te_t)


if __name__ == "__main__":
    # Example
    Ni = 4  # Number of external inputs
    Nh = 5  # Number of hidden units
    No = 1  # Number of output units
    alpha_i = 0.0  # Input weight decay
    alpha_o = 0.0  # Ouput weight decay
    max_iter = 500  # Maximum number of iterations
    eta = 0.001  # Gradient decent parameter
    t_Nh = 2  # Number of hidden units in TEACHER net
    noise = 1.0  # Relative amplitude of additive noise
    ptrain = 100  # Number of training examples
    ptest = 100  # Number of test examples
    I_gr = 1  # Inital max gradient iterations
    range = 0.5  # Inital weight range

    tr_i, tr_t, te_i, te_t = getdata(Ni, t_Nh, No, ptrain, ptest, noise)

