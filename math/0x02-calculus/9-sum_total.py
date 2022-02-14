#!/usr/bin/env python3
"""function for summation"""


def summation_i_squared(n):
    """returns summation of i**2 to n"""
    if n < 1:
        return None
    elif n == 1:
        return 1
    else:
        return (n**2 + summation_i_squared(n-1))
