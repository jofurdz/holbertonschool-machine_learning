#!/usr/bin/env python3
"""function for testing neural network"""
import tensorflow.leras as K


def test_model(network, data, labels, verbose=True):
    """tests neural network"""
    return network.test(x=data, y=labels, verbose=verbose)
