#!/usr/bin/env python3
"""function for convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """returns ndarray of convolved images"""
    h, w, m = images.shape[1], images.shape[2], images.shape[0]
    kh, kw = kernel.shape[0], kernel.shape[1]

    convo = np.zeros((m, h - kh + 1, w - kw + 1))
    for x in range(convo.shape[1]):
        for y in range(convo.shape[2]):
            output = np.sum(images[:, x: x + kh, y: y + kw] * kernel,
                            axis=(1, 2))
            convo[:, x, y] = output
    return convo
