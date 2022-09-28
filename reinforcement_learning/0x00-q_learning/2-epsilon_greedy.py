#!/usr/bin/env python3
"""module containing epsilon_greedy"""
import numpy as np
import gym
import time
import random


def epsilon_greedy(Q, state, epsilon):
    """uses epsilon greedy function to determine action"""
    p = np.random.uniform(0, 1)

    if p < epsilon:
        next = np.random.randint(Q.shape[1])
    else:
        next = np.argmax(Q[state, :])
    return next
