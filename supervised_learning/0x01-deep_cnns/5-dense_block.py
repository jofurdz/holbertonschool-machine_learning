#!/usr/bin/env python3
"""function for creating dense block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds dense block"""
    init = K.initializers.he_normal()

    for layer in range(layers):
        b1 = K.layers.BatchNormalization(axis=3)(X)
        a1 = K.layers.Activation('relu')(b1)
        c1 = K.layers.Conv2D(filters=4 * growth_rate, padding='same',
                             kernel_initializer=init, strides=(1 1),
                             kernel_size=(1, 1))(a1)
        b2 = K.layers.BatchNormalization(axis=3)(c1)
        a2 = K.layers.Activation('relu')(b2)
        c2 = K.layers.Conv2D(filters=growth_rate, padding='same',
                             kernel_size=(3, 3), strides=(1, 1),
                             kernel_initializer=init)(a2)
        X = K.layers.concatenate([X, c2])
        nb_filters += growth_rate

    return X, nb_filters
