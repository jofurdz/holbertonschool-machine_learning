#!/usr/bin/env python3
"""function for calculating moving average"""
import numpy as np


def moving_average(data, beta):
    """calculates moving average of data set"""
    value = 0
    mov_avg = []
    for x in range(len(data)):
        value = (beta * value) + ((1 - beta) * data[x])
        mov_avg.append(value / (1 - (beta ** (x + 1))))
    return (mov_avg)
