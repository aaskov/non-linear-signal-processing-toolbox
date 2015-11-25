# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from nsp_load_data import get_svm_synthetic
from nsp_svm_train import svm_train
from nsp_svm_classify import svm_classify

# Parameters
plotting = True

# Load data set
N = 200
Train, train_t, Test, test_t = get_svm_synthetic(N)

# Plot data set
if plotting:
    plt.figure(1)    
    plt.plot(Train[0:N, 0], Train[0:N, 1], '.b', label="Class 1")
    plt.plot(Train[N:, 0], Train[N:, 1], '.r', label="Class 2")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid('on')


# Train the SVM with fixed sigma and C
sigma = 1; C = 1000
SVM = svm_train(Train, train_t, sigma, C)

# Classify
prediction = svm_classify(Test, SVM)

# Plot result
if plotting:
    plt.figure(2)
    plt.plot(Test[0:N, 0], Test[0:N, 1], '.b', label="Class 1")
    plt.plot(Test[N:, 0], Test[N:, 1], '.r', label="Class 2")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid('on')
    plt.title('SVM Prediction on test set')

    for i in range(len(prediction)):
        if prediction[i] < 0:
            plt.plot(Test[i,0], Test[i,1], 'ob', fillstyle="none")
        else:
            plt.plot(Test[i,0], Test[i,1], 'or', fillstyle="none")
