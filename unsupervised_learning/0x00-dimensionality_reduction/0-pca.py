#!/usr/bin/env python3
"""function that performs principal component analysis on dataset"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on dataset"""
    U, vals, eig = np.linalg.svd(X)

    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    eig = (eig.T)[:, idx]

    var_explained = []
    eig_sum = vals.sum()

    for i in range(vals.shape[0]):
        var_explained.append(vals[i]/eig_sum)

    # Cumulative sum
    Csum = np.cumsum(var_explained)

    for i in range(Csum.shape[0]):
        if Csum[i] >= var:
            return eig[:, :i+1]

    return eig
