#!/usr/bin/env python3
"""function for building a sparse autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """creating a sparse autoencoder"""
    input = keras.layers.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(input)
    sparse = keras.regularizers.l1(lambtha)

    for x in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[x],
                                     activation='relu')(encoded)

    encoded = keras.layers.Dense(latent_dims,
                                 activation='relu',
                                 activity_regularizer=sparse)(encoded)
    decoded = keras.layers.Input(shape=(latent_dims,))
    decodedInput = decoded

    for x in range(len(hidden_layers)-1, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[x],
                                     activation='relu')(decoded)

    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    encoder = keras.models.Model(input, encoded)
    decoder = keras.models.Model(decodedInput, decoded)

    autoencoder = keras.models.Model(input, decoder(encoder(input)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
