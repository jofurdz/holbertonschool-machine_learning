#!/usr/bin/env python3
"""concatenates two matrices across a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two  matrices"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            result = []
            for row in mat1:
                result.append(row.copy())
            for j in range(len(mat2)):
                result.append(mat2[j].copy())
            return result
    else:
        if len(mat1) != len(mat2):
            return None
        else:
            newMatrix = []
            for x in range(len(mat1)):
                newMatrix.append(mat1[x] + mat2[x])
            return newMatrix
