import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader


class Network(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x


