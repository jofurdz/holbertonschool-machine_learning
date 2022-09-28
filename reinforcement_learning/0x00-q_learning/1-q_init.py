#!/usr/bin/env python3
"""module containing q_init"""
import gym
import numpy as np
import random
import time


def q_init(env):
    """initializes q-table"""
    ass = env.action_space.n
    sss = env.observation_space.n

    qTable = np.zeros((sss, ass))
    return qTable
