#!/usr/bin/env python3
"""function for saving and loading weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves weights"""
    network.save_weights(filepath=filename, save_format=save_format)


def load_weights(network, filename):
    """laods weights"""
    return network.load_weights(filepath=filename)
