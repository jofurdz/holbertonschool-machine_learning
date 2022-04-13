#!/usr/bin/env python3
"""performs forward prop over pooling layer"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """returns output of pooling layer"""
    m, h, w, npc = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = (h - kh) // sh + 1
    pw = (w - kw) // sw + 1
    convo = np.zeros((m, ph, pw, npc))
    if mode == 'max':
        func = np.max
    if mode == 'avg':
        func = np.average
    for x in range(ph):
        for y in range(pw):
            convo[:, x, y, :] = func(
                A_prev[:, x*sh: x*sh + kh, y*sw: y*sw + kw, :],
                axis=(1, 2))
    return convo
