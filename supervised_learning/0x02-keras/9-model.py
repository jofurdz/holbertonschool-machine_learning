#!/usr/bin/env python3
"""function for saving and loading models"""
import tensorflow.keras as K


def save_model(network, filename):
    """saves model"""
    network.save(filename)


def load_model(filename):
    """loads model"""
    return K.models.load_model(filepath=filename)
