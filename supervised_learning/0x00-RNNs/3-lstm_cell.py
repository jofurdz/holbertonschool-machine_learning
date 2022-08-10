#!/usr/bin/env python3
"""module containing class LSTMCell"""
import numpy as np


class LSTMCell():
    # creating class
    def __init__(self, i, h, o):
        # initializing class
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """softmax activation function"""
        expo = np.exp(x)
        return expo / np.sum(expo, 1, keepdims=True)

    def sigmoid(self, x):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """forward prop for lstm cell"""
        cat = np.concatenate((h_prev, x_t), axis=1)
        forget = self.sigmoid(cat @ self.Wf + self.bf)
        update = self.sigmoid(cat @ self.Wu + self.bu)
        cTanh = np.tanh(cat @ self.Wc + self.bc)
        cNext = forget * c_prev + update * cTanh
        output = self.sigmoid(cat @ self.Wo + self.bo)
        hTanh = output * np.tanh(cNext)
        y = self.softmax(hTanh @ self.Wy + self.by)

        return hTanh, cNext, y
