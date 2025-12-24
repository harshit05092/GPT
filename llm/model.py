import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Multiheadattention
from layer_norm import LayerNorm
from transformer_block import TransformerBlock
from configuration import cfg

class GPTModule(nn.Module):

  def __init__(self, cfg) -> None:
    super().__init__()
    self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # Creates token embedding matrix of size vocab_size * emb_dim
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # Creates token embedding matrix of size context_length * emb_dim
    self.drop_emb = nn.Dropout(cfg["drop_rate"]) # Sets Dropout to drop_rate

    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg)
        for _ in range(cfg["n_layers"])]
    )# Generates a list of transformer according to the number of layers.
     # If 12 then generate list to 12 transformer block stacked on each other

    self.final_norm = LayerNorm(cfg) # Set nomalization equal to layer normalization which is usually done in language models

    self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False
    ) # Final projection layer that turns each model embeddings into vocabulary logits for next token prediction

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    token_embeds = self.token_emb(in_idx) # Gives token embeddingds in shape of in_idx * emb_dim
    pos_embeds = self.pos_emb(
        torch.arange(seq_len, device=in_idx.device)
        ) # Creates one-D tensor of size seq_len that represents absolute positioin in the sequence and ensures that it is placed into the
          # right CPU/GPU
    x = token_embeds + pos_embeds # Gives the postional token embeddings
    x = self.drop_emb(x) # Drops the 10%(0.1) of the input to prevent overfitting
    x = self.trf_blocks(x) # Passes them through transformer block
    x = self.final_norm(x) # Passes them through layer normalization
    logits = self.out_head(x) # Gives the next prediction
    return logits