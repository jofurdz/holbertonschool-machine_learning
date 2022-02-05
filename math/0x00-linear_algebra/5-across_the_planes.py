#!/usr/bin/env python3
"""adds two  matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """adds two matrices elemt-wise"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        newMatrix = []
        for a, b in zip(mat1, mat2):
            current_list = []
            for x, y in zip(a, b):
                current_list.append(x + y)
            newMatrix.append(current_list)
        return newMatrix


def matrix_shape(matrix):
    """"returns shape of matrix"""
    try:
        return [len(matrix)] + matrix_shape(matrix[0])
    except Exception:
        return []
