
import torch
import torch.nn as nn

class GELU(nn.Module): # Calculates gaussian expression linear unit(GeLU)

  def __init__(self) -> None:
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))