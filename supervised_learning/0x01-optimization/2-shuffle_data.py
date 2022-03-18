#!/usr/bin/env python3
"""function for random permutation"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles data points in 2 matrices"""
    shuffie = np.random.permutation(len(X))
    return (X[shuffie], Y[shuffie])