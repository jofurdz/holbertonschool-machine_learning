#!/usr/bin/env python3
"""module containing def regular function"""
import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain"""
    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not (P > 0).all():
        return None

    eval, evec = np.linalg.eig(P.T)
    evec1 = evec[:, np.isclose(eval, 1)]
    evec1 = evec1[:, 0]
    steady = evec1 / evec1.sum()

    return np.array([steady])
