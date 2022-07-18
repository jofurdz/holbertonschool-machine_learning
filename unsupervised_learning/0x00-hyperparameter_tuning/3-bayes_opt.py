#!/usr/env python3
"""module containing class BayesianOptimization"""
import numpy as np


class BayesianOptimization():
    """creating class"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """initializing class"""
        min, max = bounds
        self.f = f
        self.gp = GaussianProcess(X_init, Y_init, l, sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        self.X_s = np.linspace(min, max).reshape(-1, 1)
