# -*- coding: utf-8 -*-
"""
This program is for simulating a markov chain by sampling from a teacher
transition matrix. The longer the simulated chain is, the closer the estimated
stationary distribution comes to the true stationary distribution. Try vary 
the simulating length 'Nmax'.
"""
import numpy as np
import matplotlib.pyplot as plt


def draw(q):
    """Get random integer from distribution with cumulative distribution q."""
    r = np.random.rand(1)
    idx = np.argmax(r < q)
    return(idx)


# Initialization
plotting = True
K = 10  # Size of codebook
n = 0  # Initial state
Nmax = 1000  # Length of simulated markov chain


"""Teacher transition matrix (normalized).
This matrix describes the transition probability of jumping between the
different states in the markov chain.
"""
A = np.random.rand(K, K)
Amean = np.repeat(1/np.sum(A, axis=1), K)
Amean = np.reshape(Amean, (K, K))
Anorm = A*Amean

"""Stationary probability distribution.
We take the dot product of the teacher matrix A and the vector [1,0,..0]
100 times to get the stationary true probability distribution of A.
"""
p = np.zeros(K)
p[0] = 1
for iter in range(100):
    p = np.dot(p, Anorm)
p_stationary = p

# Compute the cumalative distribution (for easier sampling)
Ac = np.copy(Anorm)
for j in range(K):
    Ac[j, :] = np.cumsum(Anorm[j, :])


"""Simulate a Markov chain.
We randomly draw the next state in the chain with the probability from
the cumulative distribution in 'Ac'.
"""
h = np.zeros(Nmax)
for j in range(Nmax):
    n = draw(Ac[n, :])
    h[j] = n


# Plotting
if plotting:
    plt.figure(1)
    plt.subplot(211)
    plt.title('True stationary distribution')
    plt.bar(np.arange(K), p_stationary)

    plt.subplot(212)
    plt.title('Estimated stationary distribution')
    plt.hist(h, K, normed=True, hold=True)
