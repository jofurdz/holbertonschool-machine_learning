#!/usr/bin/env python3
"""function for creating confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates confusion matrix"""
    newMatrix = np.matmul(labels.T, logits)
    return (newMatrix)
