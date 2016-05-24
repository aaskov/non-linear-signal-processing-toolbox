# -*- coding: utf-8 -*-
import numpy as np


def get_sunspot(lag=2):
    """Import the sunspot data set.

    Args:
        lag (int): The dimension of the lag space matrix (default 2).

    Yields:
        year (ndarray): A year label for the observations (with lag).\n
        reading (ndarray): The recorded sunspot activity (with lag).\n
        X (ndarray): A lag space matrix of dimension: (N-d) x d.

    Examples:
        The function returns the loaded data set with a specified lag
        
        >>> year, read, lag_matrix = get_sunspot()
    """
    
    # Open and read in the data-file
    with open('data\sunspot.dat', 'r') as data:
        read_data = data.read()
    
    # Pre-process the data set
    R = read_data.split('\n')
    D = list()
    for element in R:
        set = element.split('   ')
        feature = list()
        for each in set:
            feature.append(float(each))
        D.append(feature)
    
    # Numpy conversion
    D = np.array(D)
    year = D[lag:, 0]
    reading = D[:, 1]
    
    # Assemble lag-space matrix
    if lag > 0:
        N = len(reading) - lag
        X = list()
        for i in range(N):
            X.append(reading[i:i+lag])
        X = np.array(X)
        
        return((year, reading, X))
    
    else:
        return((year, reading))
    

def data_split(data, ratio=0.2):
    """Split data set into train/test set with a ratio procentage as test.
    This function creates a copy of the data set into a train and test set
    defined by the split ratio.
    
    Args:
        data (ndarray): The data set stored with observations as rows.\n
        ratio (float): The ratio of how large the test set is (deafult 0.2).\n
        
    Yields:
        train (ndarray): Train data set.\n
        test (ndaray): Test data set.\n
    
    Examples:
        Split the data set with 20% left as test data (known as unseen).
        
        >>> train, test = data_split(data, 0.2)
    """
    split = int(len(data) - len(data)*ratio)
    train = data[0:split, ]
    test = data[split:, ]
    return((train, test))
