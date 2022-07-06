#!/usr/bin/env python3
"""module containing function viterbi"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """calculates the most likely sequence of hidden states in a hmm"""
    if (type(Observation) is not np.ndarray or Observation.ndim != 1 or
       Observation.shape[0] == 0 or
       type(Emission) is not np.ndarray or Emission.ndim != 2 or
       type(Transition) is not np.ndarray or Transition.ndim != 2 or
       type(Initial) is not np.ndarray or len(Initial) != Transition.shape[0]):
        return None, None

    N, _ = Emission.shape
    T = Observation.size

    seq_probs = Initial * Emission[:, Observation[0]][..., np.newaxis]
    buff = np.zeros((N, T))

    for t in range(1, T):
        mat = (Emission[:, Observation[t]] * Transition.reshape(N, 1, N))
        mat = (mat.reshape(N, N) * seq_probs[:, t-1].reshape(N, 1))

        mx = np.max(mat, axis=0).reshape(N, 1)
        seq_probs = np.concatenate((seq_probs, mx), axis=1)
        buff[:, t] = np.argmax(mat, axis=0).T

    P = np.max(seq_probs[:, T-1])
    link = np.argmax(seq_probs[:, T-1])
    path = [link]

    for t in range(T - 1, 0, -1):
        idx = int(buff[link, t])
        path.append(idx)
        link = idx

    return path[::-1], P
