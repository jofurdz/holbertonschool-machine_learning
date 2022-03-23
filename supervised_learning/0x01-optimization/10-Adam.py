#!/usr/bin/env python3
"""function for gradient descent using ADAm optimizer"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """returns adam optimization"""
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
