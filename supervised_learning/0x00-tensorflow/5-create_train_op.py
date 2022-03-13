#!/usr/bin/env python3
"""function for training neural network with tensorflow"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """trains nueral network using gradient descent"""
    poopla = tf.train.GradientDescentOptimizer(alpha)
    return poopla.minimize(loss)
