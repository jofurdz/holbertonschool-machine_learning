#!/usr/bin/env python3
"""performs element-wise order of operations"""


def np_elementwise(mat1, mat2):
    """does element wise order of operations"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
