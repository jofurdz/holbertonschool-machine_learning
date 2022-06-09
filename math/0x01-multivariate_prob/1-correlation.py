#!/usr/bin/python3
"""function for calculating correlation matrix"""
import numpy as np


def correlation(C):
    """calculates correlation matrix"""
    variances = np.diag(C)
    stddevs = np.sqrt(variances)
    # divide horizontally, then vertically
    return (C / stddevs) / stddevs[:, np.newaxis]
