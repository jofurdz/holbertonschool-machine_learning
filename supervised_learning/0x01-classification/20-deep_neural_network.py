#!/usr/bin/env python3
"""function for deep nueral network"""
import numpy as np


class DeepNeuralNetwork():
    """initializes class"""
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for x in range(self.L):
            if type(layers[x]) is not int or layers[x] < 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W{}".format(x + 1)
            b = "b{}".format(x + 1)
            self.weights[b] = np.zeros((layers[x], 1))
            self.weights[w] = np.random.randn(layers[x], prev)\
                * np.sqrt(2 / prev)
            prev = layers[x]

    """getter for L"""
    @property
    def L(self):
        return(self.__L)

    """getter for cache"""
    @property
    def cache(self):
        return(self.__cache)

    """getter for weights"""
    @property
    def weights(self):
        return(self. __weights)

    def forward_prop(self, X):
        """calculates the forward propogation of deep nueral network"""
        self.__cache["A0"] = X
        for x in range(self.L):
            W = self.__weights["W{}".format(x + 1)]
            b = self.__weights["b{}".format(x + 1)]
            y = np.matmul(W, self.cache["A{}".format(x)]) + b
            A = 1 / (1 + np.exp(-y))
            self.__cache["A{}".format(x + 1)] = A
        return (A, self.__cache)

    def cost(self, Y, A):
        """calculates the cost of deep nueral network"""
        m = Y.shape[1]
        loss = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = -(1 / m) * (loss)
        return (cost)

    def evaluate(self, X, Y):
        """evaluates the deep nueral network's predictions"""
        pred, cache = self.forward_prop(X)
        cost = self.cost(Y, pred)
        pred = np.where(pred >= 0.5, 1, 0)
        return (pred, cost)
