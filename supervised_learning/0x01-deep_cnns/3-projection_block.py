#!/usr/bin/env python3
"""function for building a projection block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """builds projection block"""
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    c1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                         kernel_initializer=init)(A_prev)
    b1 = K.layers.BatchNormalization(axis=3)(c1)
    a1 = K.layers.Activation('relu')(b1)
    c2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                         kernel_initializer=init)(a1)
    b2 = K.layers.BatchNormalization(axis=3)(c2)
    a2 = K.layers.Activation('relu')(b2)
    c3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                         kernel_initializer=init)(a2)
    b3 = K.layers.BatchNormalization(axis=3)(c3)
    c4 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                         kernel_initializer=init)(A_prev)
    b4 = K.layers.BatchNormalization(axis=3)(c4)
    add = K.layers.Add()([b3, b4])
    return K.layers.Activation('relu')(add)
