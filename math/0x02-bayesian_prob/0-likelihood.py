#!/usr/bin/env python3
"""function for calculating probability using binomial distribution"""
import numpy as np


def ncr(n, r):
    """calculates combination"""
    nFac = np.math.factorial(n)
    rFac = np.math.factorial(r)

    return nFac / (rFac * np.math.factorial(n - r))


def likelihood(x, n, P):
    """calculates the probability using bayesian distribution"""
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x <= 0:
        raise ValueError("x must be an integer\
         that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray:
        raise TypeError("P must be a 1D numpy.ndarray")
    return ncr(n, x) * (P ** x) * ((1 - P) ** (n - x))
