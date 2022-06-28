#!/usr/bin/env python3
"""function for calculating maximization step in the EM"""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM"""
    if (type(X) is not np.ndarray or X.ndim != 2 or
       type(g) is not np.ndarray or g.ndim != 2 or
       X.shape[0] != g.shape[1] or
       not np.isclose(np.sum(g, axis=0), np.ones(X.shape[0],)).all()):
        return None, None, None

    k, n = g.shape
    d = X.shape[1]

    pi = g.sum(axis=1)
    pi /= n
    m = np.dot(g, X)
    m /= g.sum(1)[:, None]
    S = np.zeros((k, d, d))

    for i in range(k):
        ys = X - m[i, :]
        S[i] = (
            g[i, :, None, None] * np.matmul(ys[:, :, None], ys[:, None, :])
        ).sum(axis=0)
    S /= g.sum(axis=1)[:, None, None]

    return pi, m, S
