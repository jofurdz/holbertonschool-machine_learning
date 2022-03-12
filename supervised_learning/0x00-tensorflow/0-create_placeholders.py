#!/usr/bin/env python3
"""function to create placeholders for the nueral network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """creates placeholders"""
    x = tf.placeholder("float", shape=(None, nx), name='x')
    y = tf.placeholder("float", shape=(None, classes), name='y')
    return (x, y)
