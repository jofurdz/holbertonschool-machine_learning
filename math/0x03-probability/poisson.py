#!/usr/bin/env python3
"""represents a poisson distribution"""


class Poisson():
    """class that represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        self.lambtha = lambtha
        if not data:
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")
            else:
                self.data = lambtha
        else:
            length = len(data)
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif length < 2:
                raise ValueError("data must contain multiple values")
            else:
                total = 0
                for x in data:
                    total = x + total
                self.lambtha = float(total / length)
