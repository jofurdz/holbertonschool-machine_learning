#!/usr/bin/env python3
"""module for initializing cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    max, min = np.max(X, axis=0), np.min(X, axis=0)
    clusters = np.random.uniform(low=min, high=max, size=(5, 2))
    return clusters
