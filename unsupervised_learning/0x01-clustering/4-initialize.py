#!/usr/bin/env python3
"""module that initializes variables for gaussian mixture"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model (GMM)"""
    if (type(X) is not np.ndarray or len(X.shape) != 2 or
       type(k) is not int or k <= 0):
        return None, None, None

    n, d = X.shape
    pi = np.ones(k) / k
    m, clss = kmeans(X, k)
    S = np.ndarray((k, d, d))
    S[:] = np.identity(d)

    return pi, m, S
