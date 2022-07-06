#!/usr/bin/env python3
"""module containing funtion: absorbing"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing"""
    diag = np.diag(P)

    state1, state2 = P.shape

    # print(P.shape)
    # print(P)

    if len(P.shape) != 2:
        return None
    if (state1 != state2) and type(P) is not np.ndarray:
        return None
    if (diag == 1).all():
        return True
    if not (diag == 1).any():
        return False

    for row in range(state1):
        for col in range(state2):
            if (row == col) and (row + 1 < len(P)):
                if P[row + 1][col] == 0 and P[row][col + 1] == 0:
                    return False
    return True
