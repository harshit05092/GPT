import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration import cfg

class LayerNorm(nn.Module):

  def __init__(self, cfg) -> None:
    super().__init__()
    self.scale = nn.Parameter(torch.ones(cfg["emb_dim"])) # Trainable parameter gamma
    self.shift = nn.Parameter(torch.zeros(cfg["emb_dim"])) # Trainable parameter beta

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True) # Calculating mean of the layer
    var = x.var(dim=-1, keepdim=True, unbiased=False) # Calculating variance of the layer
    x = (x - mean) / torch.sqrt(var + 1e-5) # calculating standardized number. Dividing by 1e-5 to prevent diving by 0
    return x * self.scale + self.shift # Returning x * gamma + beta