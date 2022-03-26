#!/usr/bin/env python3
"""function for calculating sensitivity"""
import numpy as np


def sensitivity(confusion):
    """calculates sensitivy for confusion matrix"""
    positive = np.sum(confusion, axis=1)
    correctPos = np.diagonal(confusion)
    sensitivity = correctPos / positive
    return np.array(sensitivity)
