# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lin
from gp_kernels import kernel_gp, kernel_gaussian


def gp_loglik(K_test, K_test_train, K_train, test_t, train_t, sig2, beta):
    """Gaussian process log-likelihood calculation."""

    # Kernel transformation
    A = kernel_gp(K_train, sig2, beta)
    B = kernel_gp(K_test, sig2, beta)
    C = kernel_gaussian(K_test_train, sig2)

    # GP
    Q = C.dot(lin.pinv(A))
    Btt = B-Q.dot(C.T)
    gp_prediction = Q.dot(train_t)

    # Log-likelihood
    gp_dist = gp_prediction - test_t
    gp_log_test = -0.5*np.log(lin.det(Btt)) - 0.5*gp_dist.dot(lin.pinv(Btt)).dot(gp_dist)
    gp_std_test = np.sqrt(np.abs(np.diag(Btt)))

    return((gp_log_test, gp_std_test, gp_prediction))
