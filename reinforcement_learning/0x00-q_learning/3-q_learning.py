#!/usr/bin/env python3
"""module containing train function"""
import numpy as np
import gym
import time
import random


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """does Q-learning"""
    rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()

        done = False
        rewards_current_episode = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            newState, reward, done, _ = env.step(action)

            Q[state, action] = Q[state, action] * (1 - alpha) + \
                alpha * (reward + gamma * np.max(Q[newState, :]))

            state = newState
            rewards_current_episode += reward

            if done is True:
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)
        rewards.append(rewards_current_episode)

    return Q, rewards
