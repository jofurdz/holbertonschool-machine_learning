#!/usr/bin/env python3
"""function for performing pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """returns ndarray containing pooled images"""
    m, h = images.shape[0], images.shape[1]
    w, c = images.shape[2], images.shape[3]
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride[0], stride[1]
    ch = int(((h - kh) / sh) + 1)
    cw = int(((w - kw) / sw) + 1)
    convo = np.zeros((m, ch, cw, c))
    for x in range(ch):
        j = x * sh
        for y in range(cw):
            k = y * sw
            output = images[:, j:j + kh, k:k + kw, :]
            if mode == 'max':
                convo[:, x, y, :] = np.max(output, axis=(1, 2))
            if mode == 'avg':
                convo[:, x, y, :] = np.average(output, axis=(1, 2))
    return convo
