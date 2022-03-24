#!/usr/bin/env python3
"""function for calculating F1 score"""
import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates f1 score of confusion matrix"""
    sens = sensitivity(confusion)
    prec = precision(confusion)
    score = ((sens * prec) / (sens + prec))
    return (2 * score)
