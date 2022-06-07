#!/usr/bin/env python3
"""function for back prop in convolutional layers with pooling"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """returns partial derivates with respect to the previous layer"""
    m, h_new, w_new, c = dA.shape
    h_prev, w_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
