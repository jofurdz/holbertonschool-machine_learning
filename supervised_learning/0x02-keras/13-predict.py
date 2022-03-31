#!/usr/bin/env python3
"""function for predicting neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """makes prediction for neural network"""
    return network.predict(data, verbose=verbose)
