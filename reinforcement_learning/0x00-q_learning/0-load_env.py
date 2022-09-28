#!/usr/bin/env python3
"""module containing load_frozen_lake"""
import gym
import numpy as np
import random
import time


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """loads frozen lake environment"""
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
