#!/usr/bin/env python3
"""module for testing optimum number of clusters by variance"""
import numpy as np


kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if (type(kmin) is not int or kmin < 1 or
       type(kmax) is not int or kmax < 1 or
       type(iterations) is not int or iterations < 1 or
       kmin >= kmax):
        return None, None
    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        d_vars.append(variance(X, results[0][0]) - variance(X, C))

    return results, d_vars
