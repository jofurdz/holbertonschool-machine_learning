#!/usr/bin/env python3
"""function for calculating cost with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates cost of nueral network"""
    sum = 0
    for x in range(1, L + 1):
        sum += np.linalg.norm(weights.get('W' + str(x)))
    return (cost + (sum * (lambtha / (2 * m))))
