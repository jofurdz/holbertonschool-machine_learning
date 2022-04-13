#!/usr/bin/env python3
"""function for performing forward prop on convolutional layer"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """returns output of convolutional layer"""
    m, h, w, c = A_prev.shape
    kh, kw, kc, nc = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph, pw = 0, 0
    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) / 2) + 1
        pw = (((w - 1) * sw + kw - w) / 2) + 1
    pad_m = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant', constant_values=0)
    ch = (h + (2 * ph) - kh) // sh + 1
    cw = (w + (2 * pw) - kw) // sw + 1
    convo = np.zeros((m, ch, cw, nc))
    for x in range(nc):
        for y in range(ch):
            for z in range(cw):
                j = y * sh
                k = z * sw
                convo[:, y, z, x] = np.sum(np.multiply(
                    W[:, :, :, x],
                    pad_m[:, j:j+kh, k:k+kw]),
                    axis=(1, 2, 3)) + b[:, :, :, x]
    return activation(convo)
