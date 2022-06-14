#!/usr/bin/env python3
"""function for calculating intersectionality"""
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


def intersection(x, n, P, Pr):
    """calculates intersectionality"""
    summ = np.sum(Pr)
    if type(Pr) is not np.ndarray or np.shape(Pr) != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.count_nonzero((P < 0) | (P > 1)) > 0:
        raise ValueError("All values in {} must be in the range\
            [0, 1]".format(P[((P < 0) | (P > 1)).argmax()]))
    if np.count_nonzero((Pr < 0) | (Pr > 1)) > 0:
        raise ValueError("All values in {} must be in the range\
            [0, 1]".format(Pr[((Pr < 0) | (Pr > 1)).argmax()]))
    if summ != 1:
        raise ValueError("Pr must sum to 1")
    return (likelihood(x, n, P) * Pr)