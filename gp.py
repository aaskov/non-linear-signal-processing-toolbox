# -*- coding: utf-8 -*-
""" Example of using the Gaussian Process (GP) for function approximations.
In this scipt, we will ...

"""
import numpy as np
import matplotlib.pyplot as plt
from gp.gp_kernels import kernel
from gp.gp_loglik import gp_loglik
from data.sunspot import get_sunspot, data_split


# Parameters and settings
plotting = True
N_SIG2 = 20
N_BETA = 20
MAX_BETA = 1
MIN_BETA = 500
MAX_SIG2 = 2
MIN_SIG2 = 0.01


# Load the sunspot data set 
lag = 5
year, reading, L = get_sunspot(lag)

# Some statics (for later use)
empirical_var = np.var(L[:, 0])

# Split data set into a train/test set
Train, Test = data_split(L)
Train_t, Test_t = data_split(reading[lag:])

# Construct kernels
K_train_train = kernel(Train.T, Train.T)
K_test_train = kernel(Test.T, Train.T)
K_test_test = kernel(Test.T, Test.T)

"""Initialize the algorihm.

We will search for the optimal values of sigma^2 
and beta by a grid search method. The values of the grids has been specified 
in the 'parameters and settings' section.
"""
best_loglike = -np.inf
best_leastsq = np.inf
log_array = np.zeros((N_SIG2, N_BETA))
leastsq_array = np.zeros((N_SIG2, N_BETA))
beta_array = np.linspace(MIN_BETA, MAX_BETA, N_BETA)
sig2_array = np.linspace(MIN_SIG2, MAX_SIG2, N_SIG2)

for gg in range(N_SIG2):
    for ss in range(N_BETA):
        # Get the current values
        sig2 = sig2_array[gg]
        beta = beta_array[ss]

        # Calculate the log-likelihood for this setup
        log_test, std_test, prediction = gp_loglik(K_test_test, K_test_train,
                                                   K_train_train, Test_t,
                                                   Train_t, sig2, beta)

        # Store the log-likelihood result
        log_array[gg, ss] = log_test
        if log_test > best_loglike:
            best_loglike = log_test
            best_pred_loglike = prediction
            best_std_pred = std_test
            best_beta = beta
            best_sig2 = sig2

        # Store the least-square result
        ls = np.mean(np.power(prediction - Test_t, 2))/empirical_var
        leastsq_array[gg, ss] = ls
        if ls < best_leastsq:
            best_leastsq = ls
            best_pred_leastsq = prediction
            best_beta_ls = beta
            best_sig2_ls = sig2

# Error meassure of the best prediction
error_LL = np.mean(np.power(best_pred_loglike - Test_t, 2))/empirical_var
error_LS = np.mean(np.power(best_pred_leastsq - Test_t, 2))/empirical_var


# Plotting
if plotting:
    axis = year[len(Train):]
    plt.figure(1)
    """
    A plot of the (unseen) test data set with the best prediction results 
    based on the log-likelihood or the least-squares  measure.
    """
    plt.plot(axis, Test_t, 'r-', label="Test data")
    plt.plot(axis, best_pred_loglike, 'bo-', label="Best LL")
    plt.plot(axis, best_pred_leastsq, 'go-', label="Best LS")
    plt.grid('on')
    plt.xlabel('Year')
    plt.ylabel('Sunspot intensity')
    plt.title('Test error LL: '+str(error_LL)+' Test error LS: '+str(error_LS))
    plt.legend()

    plt.figure(2)
    """
    A second plot of the test data set and the best prediction results of the 
    least-squares fit. Furthermore, a confidence interval is shown as the 2
    times the stand. diviation. for each prediction.
    """
    plt.plot(axis, Test_t, 'r-', label="Test data")
    plt.plot(axis, best_pred_loglike, 'bo', label="Best prediction")
    plt.plot(axis, best_pred_loglike+2*best_std_pred, 'b:', label="Conf. int.")
    plt.plot(axis, best_pred_loglike-2*best_std_pred, 'b:')
    plt.grid('on')
    plt.xlabel('Year')
    plt.ylabel('Sunspot intensity')
    plt.legend()
