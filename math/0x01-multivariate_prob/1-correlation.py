#!/usr/bin/python3
"""function for calculating correlation matrix"""
import numpy as np


def correlation(C):
    """calculates correlation matrix"""
    if type(C) != np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    variances = np.diag(C)
    stddevs = np.sqrt(variances)
    # divide horizontally, then vertically
    return (C / stddevs) / stddevs[:, np.newaxis]
