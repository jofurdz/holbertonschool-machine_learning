#!/usr/bin/env python3
"""returns transpose of a 2D matrix"""

from numpy import transpose, shape


def matrix_transpose(matrix):
    newMatrix = []
    rows = len(matrix)
    columns = len(matrix[0])

    for x in range(columns):
        row = []
        for j in range(rows):
            row.append(matrix[j][x])
        newMatrix.append(row)

    return newMatrix
