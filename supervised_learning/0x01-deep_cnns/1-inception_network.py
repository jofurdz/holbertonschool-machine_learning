#!/usr/bin/env python3
"""function for creating an inception network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """builds inception network"""
    inputLayer = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), activation='relu',
                            strides=(2, 2), padding='same')(inputLayer)
    p1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(conv1)
    conv2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu',
                            padding='same')(p1)
    conv3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), activation='relu',
                            padding='same')(conv2)
    p2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(conv3)
    incepBlock1 = inception_block(p2, [64, 96, 128, 16, 32, 32])
    incepBlock2 = inception_block(incepBlock1, [128, 128, 192, 32, 96, 64])
    p3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(incepBlock2)
    incepBlock3 = inception_block(p3, [192, 96, 208, 16, 48, 64])
    incepBlock4 = inception_block(incepBlock3, [160, 112, 224, 24, 64, 64])
    incepBlock5 = inception_block(incepBlock4, [128, 128, 256, 24, 64, 64])
    incepBlock6 = inception_block(incepBlock5, [112, 144, 288, 32, 64, 64])
    incepBlock7 = inception_block(incepBlock6, [256, 160, 320, 32, 128, 128])
    p4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(incepBlock7)
    incepBlock8 = inception_block(p4, [256, 160, 320, 32, 128, 128])
    incepBlock9 = inception_block(incepBlock8, [384, 192, 384, 48, 128, 128])

    averagePool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                            padding='valid')(incepBlock9)
    drop = K.layers.Dropout(0.4)(averagePool)
    softmax = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer='he_normal')(drop)
    return K.Model(inputs=inputLayer, outputs=softmax)
