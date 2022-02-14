#!/usr/bin/env python3
"""function for summation"""


def summation_i_squared(n):
    """returns summation of i**2 to n"""
    if n <= 0 or n is None:
        return None
    else:
        return sum(map(lambda i: i**2, list(range(n + 1))))
