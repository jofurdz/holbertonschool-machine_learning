#!/usr/bin/env python3
"""generates input for Discriminator and Generator"""
import numpy as np
import torch
import torch.nn as nn


def sample_Z(mu, sigma, sampleType, dInput, gInput, mbatchSize=None):
  """creates input for the generator and discriminator"""
  if sampleType == 'D':
    return torch.normal(mu, sigma, (dInput, mbatchSize))
  if sampleType == 'G':
    return torch.randn((dInput, gInput))
  else:
    return (0)
