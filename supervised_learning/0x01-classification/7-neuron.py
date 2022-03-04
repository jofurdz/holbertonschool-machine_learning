#!/usr/bin/env python3
"""function for single neuron binary classification"""
from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt


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
        y = np.dot(self.W, X) + self.b
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
        dw = (1/ m) * np.matmul(dz, X.T)
        db = (1/m) * np.sum(dz)
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """trains the neuron by updating the private attributes __W, __b, and __A"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step > iterations or step <= 0:
                raise ValueError("step must be positive and <= iterations")
        for j in range(iterations):
            new_list = []
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if j % step == 0 or j == 0:
                cost = self.cost(Y, self.__A)
                new_list.append(cost)
                #print(self.__W, self.__b)
                print("Cost after {} iterations: {}".format(j, cost))
            if verbose is True:
                cost = self.cost(Y, self.__A)
                print("Cost after {} iterations: {}".format(j, cost))
        if graph is True:
            cost = self.cost(Y, self.__A)
            plt.plot(X, Y, "b-")
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)
