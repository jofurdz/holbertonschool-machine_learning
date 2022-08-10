#!/usr/bin/env python3
"""module containing function bi_rnn"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """does forward prop on a Bidirectional RNN"""
