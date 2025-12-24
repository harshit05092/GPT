import torch.nn as nn
from attention import Multiheadattention
from layer_norm import LayerNorm
from feed_forward import FeedForward
from configuration import cfg

class TransformerBlock(nn.Module): # TransformerBlock class is a sub-block of nn.Module

  def __init__(self, cfg) -> None:
    super().__init__() # Initializes parameters from nn.Module
    self.attn = Multiheadattention(
        cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"]
    )# Sets the attention module to multiheadattention
    self.ff = FeedForward(cfg) # first projects the emb dim into 4 * emb_dim and then applies GeLU activation and then
                               # converts 4 * emb_dim into emb_dim
    self.norm1 = LayerNorm(cfg) # Applies layer normalization
    self.norm2 = LayerNorm(cfg) # Applies layer normalization
    self.dropout = nn.Dropout(cfg["drop_rate"]) # Sets dropout rate to 0.1

  def forward(self, x):
    shortcut = x # Shortcut connection for attention blocks
    x = self.norm1(x) # Applies layer normalization
    x = self.attn(x) # Applies multi-head attention
    x = self.dropout(x) # Applies dropout rate
    x = x + shortcut # Applies shortcut connection

    shortcut = x # Shorcut connection for FeedForward in which GeLU activation is applied
    x = self.norm2(x) # Applies layer normalization
    x = self.ff(x) # first projects the emb dim into 4 * emb_dim and then applies GeLU activation and then
                   # converts 4 * emb_dim into emb_dim
    x = self.dropout(x) # Applies dropout rate
    x = x + shortcut # Applies shortcut connection
    return x