#!/usr/bin/env python3
"""function for building an inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """returns concatenated output of inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal()
    # first group
    # print("{}".format(A_prev.shape[1:4]))
    conv1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                            activation='relu',
                            kernel_initializer=init)(A_prev)
    conv2 = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), padding='same',
                            activation='relu',
                            kernel_initializer=init)(A_prev)
    conv4 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer=init)(conv2)
    conv3 = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding='same',
                            activation='relu',
                            kernel_initializer=init)(A_prev)
    # print("{}".format(A_prev.shape))
    # second group
    conv5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                            activation='relu',
                            kernel_initializer=init)(conv4)
    pl = K.layers.MaxPooling2D(pool_size=(3, 3), strides=1,
                               padding='same')(A_prev)
    conv6 = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), padding='same',
                            activation='relu',
                            kernel_initializer=init)(pl)
    # connect time
    concat = K.layers.Concatenate()([conv1, conv3, conv5, conv6])
    return concat
