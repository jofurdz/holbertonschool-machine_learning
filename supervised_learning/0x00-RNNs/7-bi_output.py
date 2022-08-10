#!/usr/bin/env python3
"""module containing class BidirectionalCell"""
import numpy as np


class BidirectionalCell():
    """creating class"""
    def __init__(self, i, h, o):
        """initializing class"""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """softmax activation function"""
        expo = np.exp(x)
        return expo / np.sum(expo, -1, keepdims=True)

    def forward(self, h_prev, x_t):
        """forward prop for BidirectionalCell"""
        cat = np.concatenate((h_prev, x_t), axis=1)
        hNext = (np.matmul(cat, self.Whf) + self.bhf)
        hNext = np.tanh(hNext)
        return hNext

    def backward(self, h_next, x_t):
        """back prop for BidirectionalCell"""
        cat = np.concatenate((h_next, x_t), axis=1)
        hBack = np.matmul(cat, self.Whb) + self.bhb
        hBack = np.tanh(hBack)
        return hBack

    def output(self, H):
        """calculates outputs for RNN"""
        Y = self.softmax(H @ self.Wy + self.by)
        return Y
