#!/usr/bin/env python3
"""function for calculating forward prop with L2 regularization"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """returns output of each layer"""
    cache = {}
    cache['A0'] = X
    for layer in range(L):
        b = weights['b{}'.format(layer + 1)]
        A = cache['A{}'.format(layer)]
        W = weights['W{}'.format(layer + 1)]
        z = np.matmul(W, A) + b
        if layer == (L - 1):
            T = (np.exp(z) / np.sum(np.exp(z), axis=0))
        else:
            T = np.tanh(z)
            poopla = np.random.binomial(n=1, p=keep_prob, size=T.shape)
            cache['D{}'.format(layer + 1)] = poopla
            T = (T * poopla) / keep_prob
        cache['A{}'.format(layer + 1)] = T
    return cache
