#!/usr/bin/env python3
"""function for building an identity block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """builds identity block"""
    init = K.initializers.he_normal(seed=None)
    F11, F3, F12 = filters

    c1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                         kernel_initializer=init)(A_prev)
    b1 = K.layers.BatchNormalization(axis=3)(c1)
    a1 = K.layers.Activation('relu')(b1)
    c2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                         kernel_initializer=init)(b1)
    b2 = K.layers.BatchNormalization(axis=3)(c2)
    a2 = K.layers.Activation('relu')(b2)
    c3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                         kernel_initializer=init)(b2)
    b3 = K.layers.BatchNormalization(axis=3)(c3)
    add = K.layers.Add()([b3, A_prev])
    poopla = K.layers.Activation('relu')(add)
    return poopla
