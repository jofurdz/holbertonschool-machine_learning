#!/usr/bin/env python3
"""function for saving and loading configs"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves config"""
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """loads config"""
    with open(filename, 'r') as f:
        config = f.read()
    return K.models.model_from_json(config)
