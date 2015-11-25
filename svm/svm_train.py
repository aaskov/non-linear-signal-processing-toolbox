# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.mlab import find
from nsp_svm_kernels import svm_kernel_rbf
from nsp_optimizer import qp_optimizer


def svm_train(X, y, sigma, C):
    """SVM train"""    
    # Scale data set (other scaling can be used instead)
    shift = np.mean(X, axis = 0)
    stdiv = np.std(X, axis = 0)
    scale = 1/stdiv
    
    # Apply scaling
    for c in range(len(X[0])):
        X[:, c] = scale[c] * (X[:,c] + shift[c])
    
    
    # Generate kernel
    kernel = svm_kernel_rbf(X.T, X.T, sigma)
    
    # Make kernel symmetric
    kernel = (kernel + kernel.T)/2  #+ np.diag(1 / (np.ones((len(y),1)) * C))
    
    # Formulate as QP as the problem  min:  1/2*x'*H*x + f'*x
    
    # Construct H: Represents the Hessian matrix
    H = y.dot(y.T) * kernel
    
    # Make H symmetric
    #H = (H+H.T)/2
    
    # Construct f: Represents the linear term 
    f = -np.ones(len(X))
    
    # Construct Aeq: Represents the linear coefficients in Aeq*x = beq
    Aeq = y.T
    
    # Construct beq: Represents the constant vector in Aeq*x = beq
    beq = 0
    
    # Construct lb: Represents the lower bounds elementwise in lb
    lb = np.zeros(len(y))
    
    # Construct ub: Represents the upper bounds elemetwise in ub
    ub = C*np.ones(len(y))
    
    # QP solver
    res = qp_optimizer(H, f, Aeq, beq, lb, ub)
    
    # The support vectors are non-zero of res.x
    reference = np.sqrt(np.spacing(1))
    support_vector_index = find(res.x > reference)
    support_vector = X[support_vector_index, :]
    
    # Hat
    alpha_hat = y[support_vector_index].T * res.x[support_vector_index]
    
    # Find the bias (several possibilities)
    max_pos = np.argmax(res.x)
    bias = y[max_pos] - np.sum(alpha_hat * kernel[support_vector_index, max_pos])
    
    # Rescale data
    for c in range(len(support_vector[0])):
        support_vector[:,c] = (support_vector[:,c]/scale[c]) - shift[c]
    
    # Construct a dictonary
    SVM = {}
    SVM['support_vector'] = support_vector
    SVM['support_vector_index'] = support_vector_index
    SVM['sigma'] = sigma
    SVM['C'] = C
    SVM['alpha_hat'] = alpha_hat
    SVM['shift'] = shift
    SVM['scale'] = scale
    SVM['bias'] = bias
    
    return(SVM)