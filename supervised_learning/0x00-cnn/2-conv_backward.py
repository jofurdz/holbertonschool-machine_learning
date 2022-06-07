#!/usr/bin/env python3
"""function for back prop over a convolutional layer"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """returns partial derivative with respect to the previous layer"""
    m, h_new, w_new, c_new = dZ.shape
    h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = W.shape
    sh, sw = stride
    if padding is 'valid':
        ph = 0
        pw = 0
    if padding is 'same':
