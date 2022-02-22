#!/usr/bin/env python3
"""function for binomial distrubution"""


class Binomial():
    """initializes class Binomial"""
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            elif p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = n
                self.p = p
        else:
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            elif type(data) is not list:
                raise TypeError("data must be a list")
            else:
                mean = float(sum(data) / len(data))
                dev = [(i - mean) ** 2 for i in data]
                vari = sum(dev) / len(data)
                q = vari / mean
                p = 1 - q
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p
