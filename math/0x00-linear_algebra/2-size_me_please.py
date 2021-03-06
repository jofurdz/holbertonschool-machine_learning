#!/usr/bin/env python3
"""determines shape of matrix"""


def matrix_shape(matrix):
    """"returns shape of matrix"""
    try:
        return [len(matrix)] + matrix_shape(matrix[0])
    except Exception:
        return []
