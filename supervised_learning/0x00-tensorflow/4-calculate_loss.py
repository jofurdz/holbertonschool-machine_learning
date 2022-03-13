#!/usr/bin/env python3
"""function for calculating loss using tensorflow"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """calculates loss of nueral network"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
