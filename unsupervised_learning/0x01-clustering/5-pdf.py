#!/usr/bin/env python3
"""function for calculating the probability density function"""
import numpy as np


def pdf(X, m, S):
    """Calculates the PDF of a Gaussian distribution"""
    if (type(X) is not np.ndarray or len(X.shape) != 2 or
       type(m) is not np.ndarray or m.shape[0] != X.shape[1] or
       type(S) is not np.ndarray or len(S.shape) != 2 or
       S.shape[0] != X.shape[1] or S.shape[0] != S.shape[1]):
        return None

    n, d = X.shape
    X_meaned = X - m[None, :]

    det = np.linalg.det(S)
    inverse = np.linalg.inv(S)
    norm = 1 / (np.sqrt((((2 * np.pi) ** d)) * det))
    res = np.exp(-0.5 * np.sum(((X_meaned @ inverse) * X_meaned), axis=1))
    pdf = (norm * res)
    P = np.maximum(pdf, 1e-300)

    return P
