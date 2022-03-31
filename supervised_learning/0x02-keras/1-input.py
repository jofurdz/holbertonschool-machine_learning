#!/usr/bin/env python3
"""Function for building a network with Keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds model with the Keras library"""
    reg = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    outputs = inputs
    layCount = len(layers)
    for layer in range(layCount):
        if layer is 0:
            outputs = (K.layers.Dense(
                       units=layers[layer],
                       activation=activations[1],
                       kernel_regularizer=reg
                       )(inputs))
        else:
            outputs = (K.layers.Dense(
                       units=layers[layer],
                       activation=activations[1],
                       kernel_regularizer=reg
                       )(outputs))
        if layer < layCount - 1:
            outputs = (K.layers.Dropout(1 - keep_prob))(outputs)
    return K.Model(inputs=inputs, outputs=outputs)
