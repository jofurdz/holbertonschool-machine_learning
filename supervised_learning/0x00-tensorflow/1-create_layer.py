#!/usr/bin/env python3
import tensorflow as tf
"""function for creating layers for nueral network"""


def create_layer(prev, n, activation):
    """creates layers for nueral network"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.Dense(n, activation, kernel_initializer=w, name='layer')
