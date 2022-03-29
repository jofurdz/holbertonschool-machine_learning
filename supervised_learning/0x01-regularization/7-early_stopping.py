#!/usr/bin/env python3
"""function for early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """determines if the network should be stopped early"""
    if (cost < opt_cost - threshold):
        count = 0
    else:
        count += 1
    if count < patience:
        return False, count
    else:
        return True, count
