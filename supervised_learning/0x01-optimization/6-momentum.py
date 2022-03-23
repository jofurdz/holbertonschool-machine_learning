#!/usr/bin/env python3
"""function for gradient descent using momentum optimization"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """returns momentum optimization"""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
