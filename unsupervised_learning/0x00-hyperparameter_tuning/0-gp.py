#!/usr/bin/env python3
"""module containing class GaussianProcess"""
import numpy as np


class GaussianProcess():
    """creates class GaussianProcess"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """initializes class GaussianProcess"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """calculates the covariance matrix between two matrices"""
        K = np.exp(-((X1 - X2.T)**2) / (2 * (self.l**2)))
        K = (K * self.sigma_f**2)
        return K
