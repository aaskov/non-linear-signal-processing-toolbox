# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from data.synthetic import get_svm_synthetic
from svm.svm_train import svm_train
from svm.svm_classify import svm_classify

# Parameters and settings
plotting = True

# Load synthetic data set
N = 200
Train, train_t, Test, test_t = get_svm_synthetic(N)

# Plot the data set
if plotting:
    """
    This figure shows the synthetic 2-class dataset. Notice how its impossible
    to formulate a linear discriminant between the two classes.
    """
    plt.figure(1)
    plt.plot(Train[0:N, 0], Train[0:N, 1], '.b', label="Class 1")
    plt.plot(Train[N:, 0], Train[N:, 1], '.r', label="Class 2")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid('on')

# Train the SVM with fixed sigma and C
sigma = 1
C = 1000
SVM = svm_train(Train, train_t, sigma, C)

# Classify the test set
prediction = svm_classify(Test, SVM)

# Plot the result
if plotting:
    """
    This second figure shows the test data set and the predicted labels for
    each observation. As a result of the SVM, we are able to separate the two
    classes (by a hyperplane).
    """
    plt.figure(2)
    plt.plot(Test[0:N, 0], Test[0:N, 1], '.b', label="Class 1")
    plt.plot(Test[N:, 0], Test[N:, 1], '.r', label="Class 2")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid('on')
    plt.title('SVM Prediction on test set')

    # Giving each data point the prediction label (as a specific color ring)
    for i in range(len(prediction)):
        if prediction[i] < 0:
            plt.plot(Test[i, 0], Test[i, 1], 'ob', fillstyle="none")
        else:
            plt.plot(Test[i, 0], Test[i, 1], 'or', fillstyle="none")
