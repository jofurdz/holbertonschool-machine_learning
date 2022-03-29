#!/usr/bin/env python3
"""function for calculating cost using L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost):
    """calculates cost of network using L2 reularization"""
    return cost + tf.losses.get_regularization_losses()
