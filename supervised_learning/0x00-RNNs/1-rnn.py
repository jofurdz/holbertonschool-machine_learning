#!/usr/bin/env python3
"""module containing rnn function"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """performs forward propogation for simple RNN"""
    H, Y, h_next = [h_0], [], h_0

    for x in X:
        h_next, y = rnn_cell.forward(h_next, x)
        H.append(h_next), Y.append(y)

    return np.stack(H), np.stack(Y)
