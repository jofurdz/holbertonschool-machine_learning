#!/usr/bin/env python3
"""function for creating a layer with dropout reularization"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """returns output of new layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop = tf.layers.Dropout(keep_prob)
    newLayer = tf.layers.Dense(units=n, activation=activation,
                               kernel_initializer=init,
                               kernel_regularizer=drop)
    return newLayer(prev)
