# -*- coding: utf-8 -*-
import numpy as np


def get_svm_synthetic(N=100):
    """Create a 2 class synthetic data set for testing SVM.
    
    Args:
        N (int): The number of data points (default 100).
        
    Yields:
        train (ndarray): Train data set.\n
        train_class (ndarray): Labels for train data set.\n
        test (ndarray): Test data set.\n
        test_class (ndarray): Labels for the test data set.
        
    Examples:
        This returns a data (train and test) set for 200 data points.
        
        >>> train, train_class, test, test_class = get_svm_synthetic(200)
    
    """
    class_off = 1
    
    # Train data set
    r = np.random.uniform(size=N)
    t = 2*np.pi*np.random.uniform(size=N)
    data_1 = np.array([r*np.cos(t), r*np.sin(t)])
    
    r = np.random.uniform(size=N) + class_off
    t = 2*np.pi*np.random.uniform(size=N)
    data_2 = np.array([r*np.cos(t), r*np.sin(t)])
    
    train = np.concatenate((data_1, data_2), axis=1)
    train_class = np.ones((2*N, 1))
    train_class[0:N] = -1
    
    # Test data set
    r = np.random.uniform(size=N)
    t = 2*np.pi*np.random.uniform(size=N)
    data_1 = np.array([r*np.cos(t), r*np.sin(t)])
    
    r = np.random.uniform(size=N) + class_off
    t = 2*np.pi*np.random.uniform(size=N)
    data_2 = np.array([r*np.cos(t), r*np.sin(t)])
    
    test = np.concatenate((data_1, data_2), axis=1)
    test_class = np.ones((2*N, 1))
    test_class[0:N] = -1
    
    return((train.T, train_class, test.T, test_class))
