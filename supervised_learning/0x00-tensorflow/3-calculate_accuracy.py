#!/usr/bin/env python3
"""function for calculating accuracy with tensorflow"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    yMax = tf.argmax(y, axis=1)
    predMax = tf.argmax(y_pred, axis=1)
    samesies = tf.equal(predMax, yMax)
    return tf.reduce_mean(tf.cast(samesies, tf.float32))
