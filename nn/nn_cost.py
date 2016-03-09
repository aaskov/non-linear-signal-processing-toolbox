# -*- coding: utf-8 -*-
"""
Distributed memory
"""
import numpy as np
from nn_forward import nn_forward
from nn_norm import weight_norm


def nn_cost_quad(Wi, Wo, input, target):
    """ Calculate the value of the quadratic cost function,
    i.e. 0.5*(sum of squared error)
    """
    # Calculate network outputs for all exaples
    Vj, yj = nn_forward(Wi, Wo, input)

    # Calculate the deviations from desired outputs
    ej = target - yj

    # Calculate the sum of squared errors
    error = 0.5 * np.sum(np.sum(np.power(ej, 2)))
    return error


def nn_cost_quad_decay(Wi, Wo, input, target, alpha_i=1.0, alpha_o=1.0):
    """ Calculate the value of the quadratic cost function with a weigth decay,
    i.e. 0.5*(sum of squared error) + 0.5*alpha*norm(weight)
    """
    # Calculate network outputs for all exaples
    Vj, yj = nn_forward(Wi, Wo, input)

    # Calculate the deviations from desired outputs
    ej = target - yj

    # Calculate the sum of squared errors
    error = 0.5 * np.sum(np.sum(np.power(ej, 2))) + 0.5*alpha_i*weight_norm(Wi) + 0.5*alpha_o*weight_norm(Wo)
    return error

