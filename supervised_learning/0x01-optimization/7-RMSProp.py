#!/usr/bin/env python3
"""function for gradient descent using RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates variables using RMSProp"""
    dw = beta2 * s + (1 - beta2) * (grad ** 2)
    newVar = var - (alpha * (grad / ((dw ** (0.5) + epsilon))))
    return (newVar, dw)
