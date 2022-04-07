#!/usr/bin/env python3
"""function for performing convolution on greyscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """returns ndarray of convolved image"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]
    if type(padding) is tuple:
        ph, pw = padding[0]
    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    padSize = ((0, 0), (ph, ph), (pw, pw))
    pad_m = np.pad(images, pad_width=padSize, mode='constant',
                   constant_values=0)
    ch = int((pad_m.shape[1] - kh) / sh + 1)
    cw = int((pad_m.shape[2] - kw) / sw + 1)
    convo = np.zeros((m, ch, cw))
    k = 0
    for x in range(cw):
        j = x * sh
        for y in range(ch):
            k = y * sw
            output = pad_m[:, j:j + kh, k:k + kw]
            convo[:, x, y] = np.sum(np.multiply(output, kernel), axis=(1, 2))
    return convo
