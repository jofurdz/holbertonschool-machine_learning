#!/usr/bin/env python3
"""function for determining the cofactor of a matrix"""


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
    if matrix and matrix[0] and type(matrix) is list \
            and all(type(sub) is list for sub in matrix):
        width = len(matrix)
        if matrix == [[]]:
            raise ValueError("matrix must be a non-empty square matrix")
        for height in matrix:
            if width != len(height):
                raise ValueError("matrix must be a non-empty square matrix")
        if width == 1:
            return [[1]]
        elif width == 2:
            return ([[int(matrix[1][1]), int(matrix[1][0])],
                    [int(matrix[0][1]), int(matrix[0][0])]])
        else:
            newMatrix = deepcopy(matrix)
            for i in range(len(matrix[0])):
                for j in range(len(matrix[0])):
                    subMatrix = []
                    for row in range(len(matrix[0])):
                        subMatrix.append([])
                        for column in range(len(matrix[0])):
                            if column != j and row != i:
                                subMatrix[row].append(int(matrix[row][column]))
                                # print(newMatrix[row][column], end=' ')
                """            subMatrix[row].append(matrix[row][column])
                    subMatrix.append([])
                subMatrix.pop(0)
                subMatrix.pop()
                if i == 0 or i%2 == 0:
                    det += (matrix[0][i] * determinant(subMatrix))
                else:
                    det -= (matrix[0][i] * determinant(subMatrix))
                #print(subMatrix)"""
            # return det
            subMatrix = [x for x in subMatrix if x != []]
            newMatrix[i][j] = determinant(subMatrix)
        return newMatrix
    else:
        raise TypeError("matrix must be a list of lists")


def cofactor(matrix):
    cofactor = minor(matrix)

    for row in range(len(cofactor)):
        for column in range(len(cofactor)):
            cofactor[row][column] *= pow(-1, row+column)

    return cofactor
