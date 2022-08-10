#!/usr/bin/env python3
"""module containign class GRUCell"""
import numpy as np


class GRUCell():
    """creating class GRUCell"""
    def __init__(self, i, h, o):
        """initializing class"""
        self.Wz = np.random.randn(i+h, h)
        self.Wr = np.random.randn(i+h, h)
        self.Wh = np.random.randn(i+h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """softmax activation function"""
        expo = np.exp(x)
        return expo / np.sum(expo, 1, keepdims=True)

    def forward(self, h_prev, x_t):
        """forward propogation for GRUCell"""
        cat = np.concatenate((h_prev, x_t), axis=1)
        # reset gate
        r_t = self.sigmoid(cat @ self.Wz + self.bz)
        # update gate
        z_t = self.sigmoid(cat @ self.Wr + self.br)
        z_prev = r_t * h_prev
        cat2 = np.concatenate((z_prev, x_t), axis=1)
        r_prev = np.matmul(cat2, self.Wh)
        h_t = np.tanh(r_prev + self.bh)
        h_next = ((1 - z_t) * h_prev) + (z_t * h_t)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
