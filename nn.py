# -*- coding: utf-8 -*-
"""
Neural network - nsp
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from nn.nn_gradient import nn_gradient
from nn.nn_cost import nn_cost_quad
from nn.nn_norm import two_norm
from nn.nn_forward import nn_forward
from nn.data_sunspot import get_sunspot


class NeuralNetwork():
    """
    Neural network.

    This is an implementation of a artificial neural network used in supervised
    machine learning methods. This code is based on Lars Kai Hansen et. al.
    implementation for the 02457 course.

    """
    def __init__(self, structure, train_in, train_tar, test_in, test_tar, max_iter=1000):
        """Initialize the network"""
        self.Ni = structure[0]      # Number of external inputs
        self.Nh = structure[1]      # Number of hidden units
        self.No = structure[2]      # Number of output units
        self.alpha_i = 0.0          # Input weight decay
        self.alpha_o = 0.0          # Ouput weight decay
        self.max_iter = max_iter    # Maximum number of iterations
        self.eta = 0.001            # Gradient decent parameter
        self.t_Nh = 2               # Number of hidden units in TEACHER net
        self.noise = 1.0            # Relative amplitude of additive noise
        self.I_gr = 1               # Inital max gradient iterations
        self.range = 0.2            # Inital weight range

        # Data
        self.train_input = train_in
        self.train_target = train_tar
        self.test_input = test_in
        self.test_target = test_tar

    def predict(self, input):
        """Prediction"""
        Vj, yj = nn_forward(self.Wi, self.Wo, input)
        return yj

    def train(self):
        """Train the network ased on a method"""
        # Define shapes
        ptrain = self.train_target.shape[0]
        ptest = self.test_target.shape[0]

        # Compute the signal variance for normalization
        signal_var = self.train_target - np.mean(self.train_target)
        signal_var = np.sum(signal_var*signal_var)/len(self.train_input)
        errnorm = 2/signal_var

        # Initialize network weights
        self.Wi = self.range*np.random.randn(self.Nh, self.Ni+1)
        self.Wo = self.range*np.random.randn(self.No, self.Nh+1)

        # Store errors and gradient
        self.norm_gradient = np.ones((self.max_iter, 1))
        self.e_train = np.ones((self.max_iter, 1))
        self.e_test = np.ones((self.max_iter, 1))

        # Run forrest, run
        iter = 0
        while iter < self.max_iter:
            # Direction (gradient)
            dWi, dWo = nn_gradient(self.Wi, self.Wo, self.alpha_i, self.alpha_o, self.train_input, self.train_target)

            # Step (weight update)
            self.Wi = self.Wi - self.eta*dWi
            self.Wo = self.Wo - self.eta*dWo

            # Calculate error and gradient
            self.norm_gradient[iter] = two_norm(dWi, dWo)
            self.e_train[iter] = errnorm*nn_cost_quad(self.Wi, self.Wo, self.train_input, self.train_target)/ptrain
            self.e_test[iter] = errnorm*nn_cost_quad(self.Wi, self.Wo, self.test_input, self.test_target)/ptest

            # Proceed
            iter += 1


if __name__ == "__main__":
    # Load data set
    lag = 2
    train_input, train_tar, test_input, test_target = get_sunspot(lag)

    # Setup an train netowrk
    nn = NeuralNetwork((lag, 8, 1),
                       train_input,
                       train_tar,
                       test_input,
                       test_target)
    nn.train()

    # Error print
    print('Minimum test MSE: ' + str(min(nn.e_test)))

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

