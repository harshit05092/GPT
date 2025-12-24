import torch.nn as nn
from activation import GELU
from configuration import cfg
class FeedForward(nn.Module):

    def __init__(self, cfg) -> None:
      super().__init__()
      self.layers = nn.Sequential(
          nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
          GELU(),
          nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
      ) # Projects the embedding dimension into embedding_dimension * 4 and then applies GeLU activation and then gives
        # the output in original emb_dim

    def forward(self, x):
        return self.layers(x)