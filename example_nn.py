# -*- coding: utf-8 -*-
"""
Neural network - nsp
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from nn.main import NeuralNetwork
from data.data_sunspot import get_sunspot


#%%
if __name__ == "__main__":
    print 'This file contains the Neural Network example'
    
    # Parameters and settings
    data_lag = 2
    N_input_units = data_lag
    N_hidden_units = 8
    N_output_units = 1
    net_fit_repeat = 5
    net_max_iter = 10000
    net_best_err = 9e9
    
    # Load the Sunspot dataset
    tr_in, tr_tar, te_in, te_tar = get_sunspot(data_lag)

    # Repeat network fit to find the one with the lowest error
    net_best = None
    for net_fit_count in range(net_fit_repeat):
        net = NeuralNetwork(structure=(N_input_units, N_hidden_units, N_output_units), train_input=tr_in, train_target=tr_tar, test_input=te_in, test_target=te_tar, max_iter=net_max_iter)
        net.train()

        if net.e_test[-1] < net_best_err:
            net_best = net
            net_best_err = net.e_test[-1]


    # Train and output error
    print 'Minimum test MSE:', str(min(net_best.e_test))

    # Evaluate error
    plt.figure('Error function')
    plt.semilogy(net_best.e_train, label="Train")
    plt.semilogy(net_best.e_test, label="Test")
    plt.legend()
    text = 'Weight range: ' + str(net_best.range) + ', Step-size: ' + str(net_best.eta)
    plt.title(text)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.grid()
    
    # Norm of error gradient wrt weight
    plt.figure('Error gradient norm')
    plt.semilogy(net_best.norm_gradient)
    plt.title('Weight gradient norm')
    plt.ylabel('Norm gradient')
    plt.xlabel('Iterations')
    plt.grid()
    
    # Prediction and test set
    plt.figure('Prediction on testset')
    y = net.predict(net.test_input)
    plt.plot(y, label="Prediction")
    plt.plot(net_best.test_target, label="True value")
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Target')
    plt.grid()
    
    # Network drawer (not implemented yet)
    #draw_network((Ni, Nh, No), N.Wi.T * 2, N.Wo.T * 2)