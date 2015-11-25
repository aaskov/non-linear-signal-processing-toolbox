# -*- coding: utf-8 -*-
import numpy as np
from nsp_svm_kernels import svm_kernel_rbf


def svm_classify(newdata, SVM):
    """SVM classify"""
    # Unpack SVM dictionary
    sv = SVM['support_vector']
    alpha_hat = SVM['alpha_hat']
    scale = SVM['scale']
    bias = SVM['bias']
    shift = SVM['shift']
    sigma = SVM['sigma']
    
    # Shift and scale data
    for c in range(len(sv[0])):
        sv[:, c] = scale[c] * (sv[:, c] + shift[c])
        
    for c in range(len(newdata[0])):
        newdata[:, c] = scale[c] * (newdata[:, c] + shift[c])
    
    # Classify new data
    f = np.dot(np.asmatrix(svm_kernel_rbf(sv.T, newdata.T, sigma)).T, alpha_hat.T) + bias
    prediction = np.sign(f)
    prediction = np.array(prediction)
    
    return(prediction)
