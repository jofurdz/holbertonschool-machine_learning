#!/usr/bin/env python3
"""performs convolution on greyscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """returns ndarray of convolved images with custom padding"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    ph, pw = padding[0], padding[1]
    padSize = ((0, 0), (ph, ph), (pw, pw))
    pad_m = np.pad(images, pad_width=padSize, mode='constant',
                   constant_values=0)
    convo = np.zeros((m, h + (2 * ph) - kh + 1, w + (2 * pw) - kw + 1))
    for x in range(convo.shape[1]):
        for y in range(convo.shape[2]):
            output = np.sum(pad_m[:, x: x + kh, y: y + kw] * kernel,
                            axis=(1, 2))
            convo[:, x, y] = output
    return convo
