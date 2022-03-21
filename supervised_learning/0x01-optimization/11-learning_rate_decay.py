#!/usr/bin/env python3
"""function that updates learning rate"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates learning rate using inverse time decay"""
    return alpha / (1 + decay_rate * int(global_step / decay_step))
