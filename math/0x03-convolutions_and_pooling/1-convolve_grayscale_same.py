#!/usr/bin/env python3
"""function for performing convultion on greyscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """returns ndarray of convolved images"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    pad_h = kh // 2
    pad_w = kw // 2
    padSize = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
    pad_m = np.pad(images, pad_width=padSize, mode='constant',
                   constant_values=0)
    convo = np.zeros((m, h, w))
    for x in range(convo.shape[1]):
        for y in range(convo.shape[2]):
            output = np.sum(pad_m[:, x: x + kh, y: y + kw] * kernel,
                            axis=(1, 2))
            convo[:, x, y] = output
    return convo
