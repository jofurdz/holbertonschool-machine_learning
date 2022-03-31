#!/usr/bin/env python3
""""function for building a network with keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds neural network using keras"""
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)
    for layer in range(len(layers)):
        model.add(K.layers.Dense(layers[layer], kernel_regularizer=reg,
                  activation=activations[layer], input_dim=nx))
        if layer < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
