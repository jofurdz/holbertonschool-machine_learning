#!/usr/bin/env python3
"""function for momentum optimization using gradient descent"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates variable using gradient descent with  momentum optimizer"""
    newMom = beta1 * v + (1 - beta1) * grad
    newVar = var - alpha * newMom
    return (newVar, newMom)
