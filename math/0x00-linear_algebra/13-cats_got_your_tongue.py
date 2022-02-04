#!/usr/bin/env python3

from hashlib import new
import numpy as np


def np_cat(mat1, mat2, axis=0):
    new_list = []
    a = np.array(mat1)
    b = np.array(mat2)
    new_list = np.concatenate((a, b), axis)
    return new_list
