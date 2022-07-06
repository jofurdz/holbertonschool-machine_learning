#!/usr/bin/env python3
"""module containing function baum_welch"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """performs baum-welch algorithm for hmm"""
