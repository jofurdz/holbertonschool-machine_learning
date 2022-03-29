#!/usr/bin/env python3
"""function for gradient descent with dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates weights using dropout regularization"""
    m = Y.shape[1]
    for layer in range(L, 0, -1):
        A = cache['A{}'.format(layer)]
        A1 = cache["A{}".format(layer - 1)]
        if layer == L:
            dz = (A - Y)
        else:
            dz1 = dz
            poopla = cache['D{}'.format(layer)]
            dz = np.matmul(W.T, dz1) * (1 - (A**2))
            dz = (dz * poopla) / keep_prob
        W = weights['W{}'.format(layer)]
        b = weights['b{}'.format(layer)]
        dw = (1 / m) * np.matmul(dz, A1.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        weights['W{}'.format(layer)] = W - (alpha * dw)
        weights['b{}'.format(layer)] = b - (alpha * db)
