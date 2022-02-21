#!/usr/bin/env python3
"""function that represents an exponential distribution"""


class Exponential():
    """class for exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """initializes class Exponential"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """returns pdf"""
        if x < 0:
            return 0
        else:
            e = 2.7182818285
            temp = self.lambtha
            return (temp * (e ** (-1 * temp * x)))

    def cdf(self, x):
        """returns cdf"""
        if x < 0:
            return 0
        else:
            e = 2.7182818285
            temp = self.lambtha
            return (1 - (e ** (-1 * temp * x)))
