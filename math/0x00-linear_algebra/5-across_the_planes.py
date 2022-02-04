#!/usr/bin/env python3
"""adds two  matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """adds two matrices elemt-wise"""
    newMatrix = []
    for a, b in zip(mat1, mat2):
        current_list = []
        for x, y in zip(a, b):
            current_list.append(x + y)
        newMatrix.append(current_list)
    return newMatrix
