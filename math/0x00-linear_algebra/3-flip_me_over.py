#!/usr/bin/env python3

from numpy import transpose, shape


def matrix_transpose(matrix):
    trans_mat = transpose(matrix)
    x = trans_mat.tolist()
    return x
