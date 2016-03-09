# -*- coding: utf-8 -*-
"""
Distributed memory
"""
import numpy as np


def two_norm(dWi, dWo):
    """Calculate the two norm of the gradient"""
    return np.sqrt(np.sum(np.sum(np.power(dWi, 2))) +
                   np.sum(np.sum(np.power(dWo, 2))))


def weight_norm(W):
    """Calculate the squared sum of weights"""
    return np.sum(np.power(W, 2))

