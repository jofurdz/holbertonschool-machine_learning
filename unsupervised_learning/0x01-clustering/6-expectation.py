#!/usr/bin/env python3
"""calculates the expectation step"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM"""
    if (type(X) is not np.ndarray or X.ndim != 2 or
       type(pi) is not np.ndarray or pi.ndim != 1 or
       type(m) is not np.ndarray or m.ndim != 2 or
       type(S) is not np.ndarray or S.ndim != 3 or
       S.shape[0] != pi.size or not np.isclose(np.sum(pi), [1])[0] or
       S.shape[1] != S.shape[2] or S.shape[1] != X.shape[1] or
       m.shape[1] != X.shape[1]):
        return None, None

    k = pi.shape[0]

    pdfs = [pdf(X, m[i], S[i]) * pi[i] for i in range(k)]

    g = np.array(pdfs)  # Probabilities
    likelihood = g.sum(axis=0)
    g /= likelihood
    loglikelihood = np.sum(np.log(likelihood))

    return g, loglikelihood
