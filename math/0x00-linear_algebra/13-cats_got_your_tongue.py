#!/usr/bin/env python3
"""function to concatenate matrices along specific axis"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates 2 matrices along specific axis"""
    new_list = []
    a = np.array(mat1)
    b = np.array(mat2)
    new_list = np.concatenate((a, b), axis)
    return new_list
