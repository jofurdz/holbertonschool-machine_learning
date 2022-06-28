#!/usr/bin/env python3
"""performs expectation maximization"""
import numpy as np
expectation = __import__('6-expectation').expectation
initialize = __import__('4-initialize').initialize
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM"""
    if (type(X) is not np.ndarray or X.ndim != 2 or
       type(k) is not int or k <= 0 or k > X.shape[0] or
       type(iterations) is not int or iterations <= 0 or
       type(tol) is not float or tol < 0 or
       type(verbose) is not bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    logL_prev = 0

    for i in range(iterations + 1):
        g, logL = expectation(X, pi, m, S)
        if g is None or logL is None or pi is None or m is None or S is None:
            return None, None, None, None, None
        if (verbose and ((i % 10 == 0 or i == iterations) or
           abs(logL - logL_prev) <= tol)):
            print('Log Likelihood after {} iterations: {}'.format(
                i, logL.round(5))
            )
        pi, m, S = maximization(X, g)
        if abs(logL - logL_prev) <= tol:
            break
        logL_prev = logL

    return pi, m, S, g, logL
