# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from data.data_synthetic import get_svm_synthetic
from svm.svm_train import svm_train
from svm.svm_classify import svm_classify

plt.rcParams['figure.figsize'] = (6, 6)  # Matplot config

#%%

if __name__ == "__main__":
    print 'This file contains the SVM example'   
    
    # Parameters and setup
    sigma = 1  # For SVM
    C = 1000  # For SVM
    N_data_points = 200 # For data
    
    
    # Load synthetic data set
    train_input, train_target, test_input, test_target = get_svm_synthetic(N_data_points)
    
    """
    Plot the data set
    
    This figure shows the synthetic 2-class dataset. Notice how its impossible
    to formulate a linear discriminant between the two classes (remark: you can
    use a little data-transformation trick, but we forget this here). 
    """
    plt.figure(1)
    plt.plot(train_input[0:N_data_points, 0], train_input[0:N_data_points, 1], '.m', label="Class 1")
    plt.plot(train_input[N_data_points:, 0], train_input[N_data_points:, 1], '.c', label="Class 2")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid('on')
    plt.title('Training dataset')
    
    
    # Train the SVM with fixed sigma and C
    SVM = svm_train(train_input, train_target, sigma, C)
    
    # Classify the test set
    prediction = svm_classify(test_input, SVM)
    
    """
    Plot the result
    
    This second figure shows the test data set and the predicted labels for
    each observation. With a SVM classifier we are able to separate the two
    classes (by a hyperplane).
    """
    plt.figure(2)
    plt.plot(test_input[0:N_data_points, 0], test_input[0:N_data_points, 1], '.m', label="Class 1")
    plt.plot(test_input[N_data_points:, 0], test_input[N_data_points:, 1], '.c', label="Class 2")
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid('on')
    plt.title('Prediction on testset (support vector shown as triangle)')
    
    # Giving each data point the prediction label (as a specific color ring)
    for i in range(len(prediction)):
        if prediction[i] < 0:
            plt.plot(test_input[i, 0], test_input[i, 1], 'om', fillstyle="none")
        else:
            plt.plot(test_input[i, 0], test_input[i, 1], 'oc', fillstyle="none")
    
    # Marking the support vectors with a square
    for idx in SVM['support_vector_index']:
        plt.plot(train_input[idx, 0], train_input[idx, 1], 'b^')
        