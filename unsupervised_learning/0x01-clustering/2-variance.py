#!/usr/bin/env python3
"""module for calculating intra-cluster variance"""
import numpy as np


def variance(X, C):
    """calculates intra-cluster variance of a dataset"""
    if (type(X) is not np.ndarray or X.ndim != 2 or
       type(C) is not np.ndarray or C.ndim != 2):
        return None

    try:
        distances = np.linalg.norm(X - np.expand_dims(C, 1), axis=-1)
        min = distances.min(axis=0)
        var = np.sum(np.square(min))

        return var

    except Exception:
        return None
