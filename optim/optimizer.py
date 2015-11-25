# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize


def qp_optimizer(H, f, Aeq, beq, lb, ub):
    """Quadratic optimizer.
    minimize:
        F = objective function     
    in matrix notation:
        F = (1/2)*x.T*H*x + c*x + c0
    
    subject to:
        Ax <= b
    """
    # Size
    N = np.shape(H)    
    
    # Minimize or maximize    
    sign = 1
    
    # Bounds
    bounds = list()
    for element in lb:
        bounds.append((lb[element], ub[element]))
    
    # Objective function
    loss = lambda x: sign * (0.5 * np.dot(x.T, np.dot(H, x))+ np.dot(f, x))
    
    jacobian = lambda x: sign * (np.dot(x.T, H) + f)
    
    cons = {'type':'ineq',
            'fun':lambda x: beq - np.dot(Aeq,x),
            'jac':lambda x: -Aeq}
    
    opt = {'disp':False}
    
    x0 = np.random.randn(N[0])
    x0[0] = 1
    
    result = optimize.minimize(loss, x0, jac=jacobian, bounds=bounds, 
                               constraints=cons, method='SLSQP', options=opt)
    
    return(result)
    
