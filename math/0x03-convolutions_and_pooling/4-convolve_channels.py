#!/usr/bin/env python3
"""function for convolution using multiple kernels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """returns ndarray of convolved images"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]
    if type(padding) is tuple:
        ph, pw = padding
    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    padSize = ((0, 0), (pw, pw), (ph, ph), (0, 0))
    pad_m = np.pad(images, pad_width=padSize, mode='constant',
                   constant_values=0)
    ch = int((pad_m.shape[1] - kh) / sh + 1)
    cw = int((pad_m.shape[2] - kw) / sw + 1)
    convo = np.zeros((m, ch, cw))
    for x in range(ch):
        j = 0
        for y in range(cw):
            convo[:, x, y] = (kernel * pad_m[:,
                                             y * sh: y * sh + kh,
                                             x * sw: x * sw + kw,
                                             :]).sum(axis=(1, 2, 3))
    return (convo)
