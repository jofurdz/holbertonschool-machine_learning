#!/usr/bin/env python3
"""module containing function backward"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    calculates backward algorithm for hmm

    Args:
        Observation (ndarray)(T,): contains the index of the observation
            T: no. observations
        Emission (ndarray)(N,M): contains the emission probability of a
            specific observation given a hidden state
            Emission[i,j]: probability of observing j given the hidden state i
            N: no. hidden states
            M: no. all possible observations
        Transition (ndarray)(N,N): transition matrix
            Transition[i,j]: probability of transitioning from the hidden
                state i to j
        Initial (ndarray)(N,1): probability of starting in a particular hidden
            state
    Returns:
        P, B // None, None if failure
        P (float): probability of obtaining the path sequence
        B (ndarray)(N,T): backward path probabilites"""
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if Observation.shape[0] == 0:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial) != Transition.shape[0]:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    B = np.zeros((N, T))
    B[:, -1] = 1

    for t in range(len(Observation) - 2, -1, -1):
        for s in range(N):
            B[s, t] = np.sum(
                B[:, t+1] * Transition[s, :] *
                Emission[:, Observation[t + 1]]
            )

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0], axis=1)[0]

    return P, B
