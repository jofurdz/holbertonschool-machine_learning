#!/usr/bin/env python3
"""function for a neural network performing binary classification"""
import numpy as np


class NeuralNetwork():
    """class for defining nueral network"""
    def __init__(self, nx, nodes):
        """initializes class"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = 0
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return(self.__W1)

    @property
    def b1(self):
        return(self.__b1)

    @property
    def A1(self):
        return(self.__A1)

    @property
    def W2(self):
        return(self.__W2)

    @property
    def b2(self):
        return(self.__b2)

    @property
    def A2(self):
        return(self.__A2)

    def forward_prop(self, X):
        """calculates the forward propogation of the nural network"""
        y1 = np.dot(self.W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-y1))
        y2 = np.dot(self.W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-y2))
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """calculates cost function for nueral network"""
        m = Y.shape[1]
        loss = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = -(1 / m) * (loss)
        return (cost)
