#!/usr/bin/env python3
"""function for class MultiNormal"""
import numpy as np


class MultiNormal():
    """creating class"""
    def __init__(self, data):
        """initializing class"""
        if type(data) is not np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean, self.cov = self.mean_cov(data.T)

    def mean_cov(X):
        """calculates the mean and covariance of a data set"""
        if type(X) is not np.ndarray and len(X.shape) != 2:
            raise TypeError("X must be a 2D numpy.ndarray")
        if X.shape[0] < 2:
            return ValueError("X must contain multiple data points")
        mean = (np.mean(X, axis=0).reshape(1, -1))
        stddev = X - mean
        cov = np.matmul(stddev.T, stddev)/(X.shape[0] - 1)
        return mean, cov
