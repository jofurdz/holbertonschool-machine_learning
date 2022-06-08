#!/usr/bin/env python3
"""function for determining the inverse of a matrix"""


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


def deepcopy(matrix):
    """copies list of lists"""
    new_matrix = []
    for row in range(len(matrix)):
        new_matrix.append([])
    for col in range(len(matrix[row])):
        new_matrix[row].append(matrix[row][col])
    return new_matrix


def minor(matrix):
    """calculates minor of matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(row) != len(matrix):
            raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []

    for row in range(len(matrix)):
        minor_row = []
        for col in range(len(matrix)):
            sub_matrix = [row.copy() for row in matrix]
            sub_matrix.pop(row)  # Remove current row index
            for i in range(len(sub_matrix)):
                sub_matrix[i].pop(col)  # Remove current col index
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix


def cofactor(matrix):
    """calculates cofactor of given matrix"""
    cofactor = minor(matrix)

    for row in range(len(cofactor)):
        for column in range(len(cofactor)):
            cofactor[row][column] *= pow(-1, row+column)

    return cofactor


def adjugate(matrix):
    """calculates adjugate of matrix"""
    tempMatrix = cofactor(matrix)
    adjugate = [[] for row in tempMatrix]
    for row in range(len(tempMatrix)):
        for column in range(len(tempMatrix)):
            adjugate[column].append(tempMatrix[row][column])
    return adjugate


def inverse(matrix):
    """function for inversing matrix"""
    invMatrix = adjugate(matrix)
    det = determinant(matrix)

    if det == 0:
        return None

    det = 1 / det

    for row in range(len(invMatrix)):
        for column in range(len(invMatrix)):
            invMatrix[row][column] *= det
    return invMatrix
