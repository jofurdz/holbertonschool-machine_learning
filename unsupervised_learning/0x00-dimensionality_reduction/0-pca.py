#!/usr/bin/env python3
"""function that performs principal component analysis on dataset"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on dataset"""
    U, Sig, V = np.linalg.svd(X)
    summy = np.sum(Sig)
    cummy = np.cumsum(Sig)

    newMat = cummy / summy

    for count, value in enumerate(newMat):
        if value >= var:
            trunc = count + 1
            print(trunc, value)
            break
    # print(np.shape(V))
    return V.T[:, :trunc]
