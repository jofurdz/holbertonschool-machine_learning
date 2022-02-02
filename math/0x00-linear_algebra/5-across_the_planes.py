#!/usr/bin/env python3

from numpy import shape, array


def add_matrices2D(mat1, mat2):
    if shape(mat1) != shape(mat2):
        return None
    else:
        vector1 = array(mat1)
        vector2 = array(mat2)
        sum_vector = vector1 + vector2
        x = sum_vector.tolist()
        return x
