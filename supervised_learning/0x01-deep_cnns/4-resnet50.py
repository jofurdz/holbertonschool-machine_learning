#!/usr/bin/env python3
"""builds ResNet-50 archictechture"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds ResNet-50 architechture"""
    init = K.initializers.he_normal()
    inputLayer = K.Input(shape=(224, 224, 3))
    c1 = K.layers.Conv2D(filters=64, kernel_size=7, padding='same',
                         strides=2, kernel_initializer=init)(inputLayer)
    b1 = K.layers.BatchNormalization(axis=3)(c1)
    a1 = K.layers.Activation('relu')(b1)
    p1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(a1)
    filters = [64, 64, 128]
    pBlock = projection_block(p1, filters, s=1)
    iBlock1 = identity_block(pBlock, filters)
    iBlock2 = identity_block(iBlock1, filters)

    filters = [128, 128, 512]
    pBlock1 = projection_block(iBlock2, filters, s=2)
    iBlock3 = identity_block(pBlock1, filters)
    iBlock4 = identity_block(iBlock3, filters)
    iBlock5 = identity_block(iBlock4, filters)

    filters = [256, 256, 1024]
    pBlock2 = projection_block(iBlock5, filters, s=2)
    iBlock6 = identity_block(pBlock2, filters)
    iBlock7 = identity_block(iBlock6, filters)
    iBlock8 = identity_block(iBlock7, filters)
    iBlock9 = identity_block(iBlock8, filters)
    iBlock10 = identity_block(iBlock9, filters)

    filters = [512, 512, 2048]
    pBlock3 = projection_block(iBlock10, filters, s=2)
    iBlock11 = identity_block(pBlock3, filters)
    iBlock12 = identity_block(iBlock11, filters)

    p2 = K.layers.AveragePooling2D(pool_size=7, strides=1,
                                   padding='same')(iBlock12)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=init)(p2)
    return K.Model(inputs=inputLayer, outputs=softmax)
