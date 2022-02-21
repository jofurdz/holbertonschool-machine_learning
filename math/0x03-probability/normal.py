#!/usr/bin/env python3
"""function that represents a normal distribution"""


class Normal():
    """initializes class Normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)
                variance = sum([((x - self.mean) ** 2) for x in data]) / len(data)
                self.stddev  = float(variance ** .5)

    def z_score(self, x):
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        return (z * self.stddev + self.mean)

    def pdf(self, x):
        e = 2.7182818285
        pie = 3.1415926536
        wumbo = (1 / (self.stddev * ((2 * pie)) ** .5))
        poopla = (e ** (-.5 * ((x - self.mean) / self.stddev) ** 2))
        return (wumbo * poopla)

    def cdf(self, x):
        
        chum = 