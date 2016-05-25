# -*- coding: utf-8 -*-
""" Example of using the Gaussian Process (GP) for function approximations.
In this scipt, we will ...
"""
import numpy as np
import matplotlib.pyplot as plt
from gp.gp_kernels import kernel
from gp.gp_loglik import gp_loglik
from data.data_sunspot import get_sunspot, data_split

#%%

if __name__ == "__main__":
    print 'This file contains the Gaussian Process example'    
    
    # Parameters and settings
    N_SIG2 = 20
    N_BETA = 20
    MAX_BETA = 1
    MIN_BETA = 500
    MAX_SIG2 = 2
    MIN_SIG2 = 0.01
    
    
    # Load the sunspot dataset 
    lag = 5
    year, reading, lag_matrix = get_sunspot(lag)
    empirical_var = np.var(lag_matrix[:, 0])  # Some statics (for later use)
    
    # Split data set into a train/test set
    train, test = data_split(lag_matrix)
    train_input = train[:,1:]
    train_target = train[:,0]
    test_input = test[:,1:]
    test_target = test[:,0]
    
    # Construct kernels
    K_train_train = kernel(train_input.T, train_input.T)
    K_test_train = kernel(test_input.T, train_input.T)
    K_test_test = kernel(test_input.T, test_input.T)
    
    """
    Initialize the algorihm.
    
    We will search for the optimal values of sigma^2 
    and beta by a grid search method. The values of the grids has been specified 
    in the 'parameters and settings' section above.
    """
    best_loglike = -np.inf
    best_leastsq = np.inf
    log_array = np.zeros((N_SIG2, N_BETA))
    leastsq_array = np.zeros((N_SIG2, N_BETA))
    beta_array = np.linspace(MIN_BETA, MAX_BETA, N_BETA)
    sig2_array = np.linspace(MIN_SIG2, MAX_SIG2, N_SIG2)
    
    for iter_sig in range(N_SIG2):
        for iter_beta in range(N_BETA):
            # Get the current values
            sig2 = sig2_array[iter_sig]
            beta = beta_array[iter_beta]
    
            # Calculate the log-likelihood for this setup
            log_test, std_test, prediction = gp_loglik(K_test_test, K_test_train, K_train_train, test_target, train_target, sig2, beta)
    
            # Store the log-likelihood result
            log_array[iter_sig, iter_beta] = log_test
            if log_test > best_loglike:
                best_loglike = log_test
                best_pred_loglike = prediction
                best_std_pred = std_test
                best_beta = beta
                best_sig2 = sig2
    
            # Store the least-square result
            ls = np.mean(np.power(prediction - test_target, 2))/empirical_var
            leastsq_array[iter_sig, iter_beta] = ls
            if ls < best_leastsq:
                best_leastsq = ls
                best_pred_leastsq = prediction
                best_beta_ls = beta
                best_sig2_ls = sig2
    
    # Error meassure of the best prediction
    error_LL = np.mean(np.power(best_pred_loglike - test_target, 2))/empirical_var
    error_LS = np.mean(np.power(best_pred_leastsq - test_target, 2))/empirical_var
    
    """
    Plot nr 1
    
    A plot of the (unseen) test data set with the best prediction results 
    based on the log-likelihood or the least-squares measure.
    """
    axis = year[len(train_input):]
    plt.figure(1)
    plt.plot(axis, test_target, 'b-', label="Test data")
    plt.plot(axis, best_pred_loglike, 'mo-', label="Best log-likelihood")
    plt.plot(axis, best_pred_leastsq, 'co-', label="Best least-square")
    plt.grid('on')
    plt.xlabel('Year')
    plt.ylabel('Sunspot intensity')
    title_text = 'Test error LL: %0.2f, Test error LS: %0.2f' % (error_LL, error_LS)
    plt.title(title_text)
    plt.legend(loc='upper left')

    
    """
    Plot nr 2
    
    A second plot of the test data set and the best prediction results of the 
    least-squares fit. Furthermore, a confidence interval is shown as the 2
    times the stand. diviation. for each prediction.
    """
    plt.figure(2)
    plt.plot(axis, test_target, 'b-', label="Test data")
    plt.plot(axis, best_pred_loglike, 'mo', label="Best prediction")
    plt.plot(axis, best_pred_loglike+2*best_std_pred, 'm:', label="95% conf. int.")
    plt.plot(axis, best_pred_loglike-2*best_std_pred, 'm:')
    plt.grid('on')
    plt.xlabel('Year')
    plt.ylabel('Sunspot intensity')
    plt.title('Prediction on test set with conf. interval')
    plt.legend(loc='upper left')
