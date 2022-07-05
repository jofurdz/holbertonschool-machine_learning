#!/usr/bin/env python3
"""module containing function forward"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    T = Observation.shape[0]
    N, M = Emission.shape

    # initializes forward probabilities to 0
    F = np.zeros((N, T))

    # looping through observations
    for row in range(len(Observation)):
        # looping through hidden states
        for col in range(N):
            if row == 0:
                F[col, 0] = Initial[col, 0] * Emission[col, Observation[row]]
            else:
                F[col, row] = np.sum(
                    F[:, row-1] * Transition[:, col] *
                    Emission[col, Observation[row]]
                )
    P = np.sum(F[:, -1])

    return P, F
