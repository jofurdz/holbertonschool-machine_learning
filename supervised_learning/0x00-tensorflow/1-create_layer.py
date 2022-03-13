#!/usr/bin/env python3
"""function for creating layers for nueral network"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """creates layers for nueral network"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lay = tf.layers.Dense(n, activation, kernel_initializer=w, name='layer')
    return lay(prev)
