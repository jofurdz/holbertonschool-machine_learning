#!/usr/bin/env python3
"""function for single neuron binary classification"""
import numpy as np


class Neuron():
    """initializes class Neuron"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    """getter for W"""
    @property
    def W(self):
        return(self.__W)

    """getter for b"""
    @property
    def b(self):
        return(self.__b)

    """getter for A"""
    @property
    def A(self):
        return(self.__A)

    def forward_prop(self, X):
        """calculates the forward propogation of the nueron"""
        y = np.matmul(self.W, X) + self.b
        self.__A = 1/(1 + np.exp(-y))
        return (self.__A)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        loss = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = -(1 / m) * (loss)
        return (cost)

    def evaluate(self, X, Y):
        """evaluates the nueron's predictions"""
        pred = self.forward_prop(X)
        cost = self.cost(Y, pred)
        limit = np.where(pred >= 0.5, 1, 0)
        return limit, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = np.subtract(A, Y)
        dw = (1 / m) * np.matmul(X, dz.T)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)
