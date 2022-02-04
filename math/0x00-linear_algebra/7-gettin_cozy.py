#!/bin/usr/env python3


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two  matrices"""
    if axis == 0:

    else:
        if len(mat1) != len(mat2):
            return None
        else:
            newMatrix = []
            for x in  range(len(mat1)):
                newMatrix.append(mat1[x] + mat2[x])
            return newMatrix