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
    
    # Load the Sunspot dataset
    lag = 2
    tr_in, tr_tar, te_in, te_tar = get_sunspot(lag)

    # Setup the network
    nn = NeuralNetwork(structure=(lag, 8, 1), train_input=tr_in, train_target=tr_tar, test_input=te_in, test_target=te_tar)

    # Train and output error
    nn.train()
    print 'Minimum test MSE:', str(min(nn.e_test))

    # Evaluate error
    plt.figure('Error function')
    plt.semilogy(nn.e_train, label="Train")
    plt.semilogy(nn.e_test, label="Test")
    plt.legend()
    text = 'Weight range: ' + str(nn.range) + ', Step-size: ' + str(nn.eta)
    plt.title(text)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.grid()

