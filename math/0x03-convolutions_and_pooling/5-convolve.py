#!/usr/bin/env python3
"""function for performing convolution with multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """returns ndarray of convolved images"""
    m, h = images.shape[0], images.shape[1]
    w, c = images.shape[2], images.shape[3]
    kh, kw = kernels.shape[0], kernels.shape[1]
    nc = kernels.shape[3]
    sh, sw = stride[0], stride[1]
    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) / 2) + 1
        pw = (((w - 1) * sw + kw - w) / 2) + 1
    if padding == 'valid':
        ph = 0
        pw = 0
    if type(padding) is tuple:
        ph = padding[0]
        pw = padding[1]
    padSize = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    pad_m = np.pad(images, pad_width=padSize, mode='constant',
                   constant_values=0)
    ch = int((pad_m.shape[1] - kh) / sh + 1)
    cw = int((pad_m.shape[2] - kw) / sw + 1)
    convo = np.zeros((m, ch, cw, nc))
    for x in range(ch):
        j = x * sh
        for y in range(cw):
            k = y * sw
            for z in range(nc):
                output = pad_m[:, j:j + kh, k:k + kw, :]
                kernel = kernels[:, :, :, z]
                convo[:, x, y, z] = np.sum(np.multiply(output, kernel),
                                           axis=(1, 2, 3))
    return convo
