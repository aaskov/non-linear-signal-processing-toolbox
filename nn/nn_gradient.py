# -*- coding: utf-8 -*-
"""
Neural network - nsp
"""
from __future__ import division
import numpy as np
from nn_forward import nn_forward


def nn_gradient(Wi, Wo, alpha_i, alpha_o, train_input, train_target):
    """Calculate the partial derivatives of the quadratic cost function wrt.
    to the weights. Derivatives of quadratic weight decay are included.

    Args:
        Wi: Matrix with input-to-hidden weights.\n
        Wo: Matrix with hidden-to-output weights.\n
        alpha_i: Weight decay parameter for input weights.\n
        alpha_o: Weight decay parameter for output  weights.\n
        train_input: Matrix with examples as rows.\n
        train_target: Matrix with target values as rows.

    Yields:
        dWi: Matrix with gradient for input weights.\n
        dWo: Matrix with gradient for output weights.

    Examples:
        Calculate gradients

        >>> dWi, dWo = nn_gradient(...)
    """
    # Determine the number of samples
    exam, inp = train_input.shape

    # =======================
    #      FORWARD PASS
    # =======================

    # Calculate hidden and output unit activations
    Vj, yj = nn_forward(Wi, Wo, train_input)

    # =======================
    #     BACKWARD PASS
    # =======================

    # Calculate derivative
    # by backpropagating the errors from the desired outputs

    # Output unit deltas
    delta_o = -(np.atleast_2d(train_target).T - yj)

    # Hidden unit deltas
    r, c = Wo.shape
    delta_h = (1.0 - np.power(Vj, 2)) * (delta_o.dot(Wo[:, :-1]))

    # Partial derivatives for the output weights
    dWo = delta_o.T.dot(np.concatenate((Vj, np.ones((exam, 1))), 1))

    # Partial derivatives for the input weights
    dWi = delta_h.T.dot(np.concatenate((train_input, np.ones((exam, 1))), 1))

    # Add derivative of the weight decay term
    dWi = dWi + alpha_i*Wi
    dWo = dWo + alpha_o*Wo

    return (dWi, dWo)


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

    # Load data
    from data_random import get_random_data
    train_input, train_target, te_input, te_target = get_random_data(Ni, t_Nh, No, ptrain, ptest, noise)

    # Initialize network weights
    Wi = range*np.random.randn(Nh, Ni+1)
    Wo = range*np.random.randn(No, Nh+1)

    # dWi, dWo = nn_gradient(Wi, Wo, alpha_i, alpha_o, train_input, train_target)

    # Determine the number of samples
    exam, inp = train_input.shape

    # ###################### #
    # #### FORWARD PASS #### #
    # ###################### #

    # Calculate hidden and output unit activations
    Vj, yj = nn_forward(Wi, Wo, train_input)

    # ###################### #
    # #### BACKWARD PASS ### #
    # ###################### #

    # Calculate derivative of
    # by backpropagating the errors from the desired outputs

    # Output unit deltas
    delta_o = -(train_target - yj)

    # Hidden unit deltas
    r, c = Wo.shape
    delta_h = (1.0 - np.power(Vj, 2)) * (delta_o.dot(Wo[:, :c-1]))

    # Partial derivatives for the output weights
    dWo = delta_o.T.dot(np.concatenate((Vj, np.ones((exam, 1))), 1))

    # Partial derivatives for the input weights
    dWi = delta_h.T.dot(np.concatenate((train_input, np.ones((exam, 1))), 1))

    # Add derivative of the weight decay term
    dWi = dWi + alpha_i*Wi
    dWo = dWo + alpha_o*Wo

