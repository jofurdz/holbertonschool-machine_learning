#!/usr/bin/env python3
"""function for building transition layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """builds transition layer"""
    init = K.initializers.he_normal()
    filter = (nb_filters * compression)

    b1 = K.layers.BatchNormalization(axis=3)(X)
    a1 = K.layers.Activation('relu')(b1)
    c1 = K.layers.Conv2D(filters=filter, padding='same',
                         strides=(1, 1), kernel_size=(1, 1),
                         kernel_initializer=init)(a1)
    p1 = K.layers.AveragePooling2D(pool_size=(2, 2), padding='same',
                                   strides=(2, 2))(c1)
    return (p1, filter)
