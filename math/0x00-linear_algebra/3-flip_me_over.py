#!/usr/bin/env python3
"""determines transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """returns transpose of 2D matrix"""
    newMatrix = []
    rows = len(matrix)
    columns = len(matrix[0])

    for x in range(columns):
        row = []
        for j in range(rows):
            row.append(matrix[j][x])
        newMatrix.append(row)

    return newMatrix
