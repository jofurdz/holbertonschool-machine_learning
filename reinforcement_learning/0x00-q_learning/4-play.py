#!/usr/bin/env python3
"""module containing play function"""
import gym
import numpy as np
import random
import time


def play(env, Q, max_steps=100):
    state = env.reset()
    done = False

    for step in range(max_steps):
        """has trained agent play an episode"""
        env.render()
        action = np.argmax(Q[state, :])

        state, reward, done, _ = env.step(action)

        if done:
            env.render()
            break
    return reward
