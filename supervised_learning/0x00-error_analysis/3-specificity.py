#!/usr/bin/env python3
"""function for calculating specificity"""
import numpy as np


def specificity(confusion):
    """calculates specificity of confusion matrix"""
    truePos = np.diagonal(confusion)
    falseNeg = np.sum(confusion, axis=1) - truePos
    falsePos = np.sum(confusion, axis=0) - truePos
    trueNeg = np.sum(confusion) - (falsePos + falseNeg + truePos)
    specific = trueNeg / trueNeg + falsePos
    return (specific)
