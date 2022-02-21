#!/usr/bin/env python3
"""function for binomial distrubution"""


from logging.config import valid_ident


class Binomial():
    """initializes class Binomial"""
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n is <= 0:
                raise ValueError("n must be a positive value")
            elif p """ask mango""":
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                """??????"""
        else:
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            elif type(data) is not list:
                raise TypeError("data must be a list")
            else:
                