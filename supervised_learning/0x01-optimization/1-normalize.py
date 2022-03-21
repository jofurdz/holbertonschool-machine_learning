#!/usr/bin/env python3
"""function for normalizing matrix"""
import numpy as np


def normalize(X, m, s):
    """normalizes matrix"""
    normal = (X - m) / s
    return (normal)
