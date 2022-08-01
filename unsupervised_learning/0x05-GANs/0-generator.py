#!/usr/bin/env python3
"""contains class Generator"""
import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
  """creating class Generator"""
  def __init__(self, input_size, hidden_size, output_size):
    """initializing class Generator"""
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, output_size),
        nn.Tanh(),
        nn.Linear(output_size, output_size)
    )

  def forward(self, x):
    """forward pass"""
    return (self.model(x))
