#!/usr/bin/env python3
"""function for creating layer with L2 reluarization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """returns output  of the new layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    newLayer = tf.layers.Dense(n, activation,
                               kernel_initializer=init,
                               kernel_regularizer=reg)
    return newLayer(prev)
