# -*- coding: utf-8 -*-
"""
Neural network - nsp
"""
import numpy as np


def nn_forward(Wi, Wo, train_inputs):
    """ Propagate exaples forward through network calculating all hidden-
    and output unit outputs.

    Args:
        Wi: Matrix with input-to-hidden weights.\n
        Wo: Matrix with hidden-to-outputs weights.\n
        train_inputs: Matrix with example inputs as rows.

    Yields:
        Vj: Matrix with hidden unit outputs as rows.\n
        yj: Vector with output unit outputs as rows.

    """
    # Determine the size of the problem
    examples, inp = train_inputs.shape

    # Calculate hidden unit outputs for every exaple
    Vj = np.concatenate((train_inputs, np.ones((examples, 1))), 1)
    Vj = Vj.dot(Wi.T)
    Vj = np.tanh(Vj)

    # Caluculate (linear) output unit outputs for every exaple
    yj = np.concatenate((Vj, np.ones((examples, 1))), 1)
    yj = yj.dot(Wo.T)

    return (Vj, yj)

