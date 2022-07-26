#!/usr/bin/env python3
"""function for variational autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """builds variantonal autoencoder"""
