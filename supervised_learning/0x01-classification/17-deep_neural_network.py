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
        return(self.__weights)
