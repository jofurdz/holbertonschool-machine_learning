#!/usr/bin/env python3
"""function for building convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """builds convolutional autoencoder"""
    input = keras.layers.Input(shape=input_dims)
    encodedInput = input
    padding = 'same'

    for x in range(len(filters)):
        encodedInput = keras.layers.Conv2D(filters[x], (3, 3),
                                           activation='relu',
                                           padding='same')(encodedInput)
        encodedInput = keras.layers.MaxPooling2D((2, 2),
                                                 padding='same')(encodedInput)

    decoded = keras.layers.Input(shape=latent_dims)
    decodedInput = decoded

    for x in range(len(filters)-1, -1, -1):
        if x == 0:
            padding = 'valid'
        decoded = keras.layers.Conv2D(filters[x], (3, 3), padding=padding,
                                      activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                                  padding='same')(decoded)
    encoder = keras.models.Model(input, encodedInput)
    decoder = keras.models.Model(decodedInput, decoded)

    autoencoder = keras.models.Model(input, decoder(encoder(input)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
