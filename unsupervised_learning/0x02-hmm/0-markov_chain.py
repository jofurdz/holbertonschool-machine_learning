#!/usr/bin/env python3
"""module containing markov_chain function"""
import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain being in a particular state
    after a specified number of iterations"""
    if (type(P) is not np.ndarray or P.ndim != 2 or
       P.shape[0] != P.shape[1] or
       type(s) is not np.ndarray or s.ndim != 2 or
       s.shape[0] != 1 or s.shape[1] != P.shape[0]):
        return None

    currentState = s.copy()

    for i in range(t):
        currentState = np.matmul(currentState, P)
    return currentState
