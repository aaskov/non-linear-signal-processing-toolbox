# -*- coding: utf-8 -*-
"""
Neural network - nsp
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from nn import NeuralNetwork
from data_sunspot import get_sunspot
from nn_diagram import draw_network


# Session
# Load data set (sunspot activity, regression)
# lag = 2
# train_input, train_tar, test_input, test_target = get_sunspot(lag)

# Load data set vestas stocks (regression)
from data_stocks import get_stock

# Lag defined input data
lag = 2
train_input, train_tar, test_input, test_target = get_stock(lag)

# Setup and train network (fully connected)
times = 10
Ni = lag
Nh = 2
No = 1

N = NeuralNetwork((Ni, Nh, No), train_input, train_tar,
                  test_input, test_target, max_iter=1000)
N.train()

for i in range(times):
    N_current = NeuralNetwork((Ni, Nh, No), train_input, train_tar,
                              test_input, test_target, max_iter=1000)
    N_current.train()

    if N_current.e_test[-1] < N.e_test[-1]:
        N = N_current

# Evaluation
print('Neural network is done training: SE = ' + str(N.e_test[-1]))

# Eval and illustrate
for i in np.arange(10):
    plt.close()

# Test and training error
plt.figure()
plt.semilogy(N.e_train, label="Train")
plt.semilogy(N.e_test, label="Test")
plt.legend()
plt.title('Init. weight range: ' + str(N.range) + ', Step-size: ' + str(N.eta))
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.grid()

# Norm of error gradient wrt weight
plt.figure()
plt.semilogy(N.norm_gradient)
plt.title('Weight gradient norm')
plt.ylabel('Norm gradient')
plt.xlabel('Iterations')
plt.grid()

# Prediction and test set
plt.figure()
y = N.predict(N.test_input)
plt.plot(y, label="Prediction")
plt.plot(N.test_target, label="True value")
plt.legend()
plt.xlabel('Year')
plt.ylabel('Target')
plt.grid()

# Network drawer
draw_network((Ni, Nh, No), N.Wi.T * 2, N.Wo.T * 2)

