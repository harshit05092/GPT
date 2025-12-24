cfg = {
    "vocab_size": 50257, # No of vocab in GPT's BPE
    "context_length": 256, # The previous context which our gpt will consider
    "emb_dim": 768, # Number of dimensions that each token will be embedded in
    "n_heads": 12,  # No of attention heads
    "n_layers": 12, # No of layers
    "drop_rate": 0.1, # rate of dropping nn randomly
    "qkv_bias": False # Bias vector for query, key, value is False
}