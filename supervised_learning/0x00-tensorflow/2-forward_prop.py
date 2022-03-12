#!/usr/bin/env python3
"""function for forward  prop using tensor flow"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """returns the forward prop of nueral network"""
    poopla = x
    for j in range(len(layer_sizes)):
        poopla = create_layer(poopla, layer_sizes[j], activations[j])
    return (poopla)
