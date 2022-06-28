#!/usr/bin/env python3
"""finds the best number of clusters for a GMM using BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if (type(kmin) is not int or kmin <= 0 or kmin > X.shape[0] or
       type(kmax) is not int or kmax <= 0 or kmax <= kmin or
       kmax > X.shape[0] or
       type(iterations) is not int or iterations <= 0 or
       type(tol) is not float or tol < 0 or
       type(verbose) is not bool):
        return None, None, None, None
