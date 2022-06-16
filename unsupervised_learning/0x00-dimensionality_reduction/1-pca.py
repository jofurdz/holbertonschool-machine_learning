#!/usr/bin/env python3
"""performs updated PCA on dataset"""
import numpy as np


def pca(X, ndim):
    """updated PCA function"""
    X_m = X - np.mean(X, axis=0)
    U, Sig, V = np.linalg.svd(X_m)
    return np.matmul(X_m, V.T[:, :ndim])
