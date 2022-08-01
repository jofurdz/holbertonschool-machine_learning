#!/usr/bin/env python3
"""contains class Discriminator"""
import numpy as np
import torch
import torch.nn as nn


class Discriminator(nn.Module):
  """creating class Discriminator"""
  def  __init__(self, input_size, hidden_size, output_size):
    """initializing class Discriminator"""
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, output_size),
        nn.Sigmoid(),
        nn.Linear(output_size, output_size)
    )

  def forward(self, x):
    """forward function"""
    return (self.model(x))
