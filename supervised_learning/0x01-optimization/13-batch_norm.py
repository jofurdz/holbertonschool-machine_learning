#!/usr/bin/env python3
"""function for batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """"normalizes an unactivated output of nueral network"""
    zNorm = ((Z - np.average(Z, axis=0)) /
             (((np.var(Z, axis=0) + epsilon) ** 0.5)))
    return (gamma * zNorm) + beta
