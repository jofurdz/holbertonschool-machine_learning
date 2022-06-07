#!/usr/bin/env python3
"""function for determining the determinant of a matrix"""


def determinant(matrix):
    """calculates the dterminant of matrix"""
    if matrix == [[]]:
        return 1
    if matrix and matrix[0] and type(matrix) is list \
            and all(type(sub) is list for sub in matrix):
        width = len(matrix)
        for height in matrix:
            if width != len(height):
                raise ValueError("matrix must be a square matrix")
        if width == 1:
            return matrix[0][0]
        elif width == 2:
            return (matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1])
        else:
            det = 0
            for i in range(len(matrix[0])):
                subMatrix = [[]]
                for row in range(len(matrix[0])):
                    for column in range(len(matrix[0])):
                        if column != i and row != 0:
                            subMatrix[row].append(matrix[row][column])
                    subMatrix.append([])
                subMatrix.pop(0)
                subMatrix.pop()
                if i == 0 or i % 2 == 0:
                    det += (matrix[0][i] * determinant(subMatrix))
                else:
                    det -= (matrix[0][i] * determinant(subMatrix))
                # print(subMatrix)
            return det
    else:
        raise TypeError("matrix must be a list of lists")
