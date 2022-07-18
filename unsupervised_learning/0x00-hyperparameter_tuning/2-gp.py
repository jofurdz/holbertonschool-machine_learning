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

    def predict(self, X_s):
        """predicts mean / standard deviation of points in a GaussianProcess"""
        K_s = self.kernel(X1=self.X, X2=X_s)
        K_s2 = self.kernel(X_s, X_s)
        inv_K = np.linalg.inv(self.K)
        mu = K_s.T.dot(inv_K).dot(self.Y).reshape(X_s.shape[0])
        sigma = np.diag(K_s2 - K_s.T.dot(inv_K).dot(K_s))
        return mu, sigma

    def update(self, X_new, Y_new):
        """updates Gaussian Process"""
        print(self.X.shape, self.Y.shape)
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
