#!/usr/bin/env python3


def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2):
        return None
    else:
        newMatrix  = []
        for x in range(len(mat1)):
            #iterate thru rows
            for j in range(len(mat2)):
                newMatrix = mat1[x][j] + mat2[x][j]

        return newMatrix
