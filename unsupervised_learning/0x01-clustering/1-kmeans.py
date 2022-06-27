#!/usr/bin/env python3
"""module for performing k-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """performs k-means on a dataset"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    n, d = X.shape

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))

    for i in range(iterations):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        clss = np.argmin(distances, axis=0)
        prev = centroids.copy()
        for j in range(k):
            if len(X[j == clss]) == 0:
                centroids[j] = np.random.uniform(low=np.min(X, axis=0),
                                                 high=np.max(X, axis=0),
                                                 size=(1, d))
            else:
                centroids[j] = np.mean(X[j == clss], axis=0)
        if np.array_equal(prev, centroids):
            break

    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    clss = np.argmin(distances, axis=0)

    return centroids, clss
