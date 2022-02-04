#!/usr/bin/env python3
"""adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    new_list = []

    if len(arr1) != len(arr2):
        return None
    else:
        for x, y in zip(arr1, arr2):
            new_list.append(x + y)
        return new_list
