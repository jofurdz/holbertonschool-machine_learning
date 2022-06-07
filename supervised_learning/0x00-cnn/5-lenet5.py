#!/usr/bin/env python3
"""function for building leNet-5 architecture"""
import tensorflow.keras as K


def lenet5(X):
    """builds LeNet-5 architecture using Keras"""
    init = K.initializers.he_normal()
    c1 = K.layers.Conv2D(6, kernel_size=(5, 5), padding='same',
                         activation='relu',
                         kernel_initializer=init)(X)
    p1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)
    c2 = K.layers.Conv2D(16, kernel_size=(5, 5), padding='valid',
                         activation='relu',
                         kernel_initializer=init)(p1)
    p2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)
    flat = K.layers.Flatten()(p2)
    layer1 = K.layers.Dense(units=120, kernel_initializer=init,
                            activation='relu')(flat)
    layer2 = K.layers.Dense(units=84, kernel_initializer=init,
                            activation='relu')(layer1)
    layer3 = K.layers.Dense(units=10, kernel_initializer=init,
                            activation='softmax')(layer2)
    newModel = K.models.Model(X, layer3)
    adam = K.optimizers.Adam()
    newModel.compile(optimizer=adam,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return newModel
