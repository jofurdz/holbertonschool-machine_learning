#!/usr/bin/env python3
"""Function for early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False):
    """trains neural network with early stopping using keras"""
    early_stop = None
    if early_stopping and validation_data:
        early_stop = [K.callbacks.EarlyStopping(patience=patience)]
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=early_stop)
