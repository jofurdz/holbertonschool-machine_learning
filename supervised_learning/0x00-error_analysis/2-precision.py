#!/usr/bin/env python3
"""function for calculating precision"""
import numpy as np


def precision(confusion):
    """calculates precision of confusion matrix"""
    positives = np.sum(confusion, axis=1)
    truePosi = np.diagonal(confusion)
    precision = truePosi / positives
    return (precision)
