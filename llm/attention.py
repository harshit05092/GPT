import torch
import torch.nn as nn
import torch.nn.functional as F

class Multiheadattention(nn.Module):

  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False) -> None:
    super().__init__() # Allow the inherited class paramenters to be initialized like backward pass and _parameters
    assert(d_out % num_heads == 0), \
    "d_out must be divisible by num_heads"
    self.d_out = d_out # Output dimensions that our transformer will output the code in
    self.num_heads = num_heads # Number of attention heads in our multihead attention layer
    self.head_dim = d_out // num_heads # Number of dimension each vector will be when then are output by each attention module
    self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias) # Weight matrix for query vector. din and dout are same generally
    self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias) # Weight matrix for key vectors
    self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias) # Weight matrix for value vectors
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout) # Dropout rate to prevent overfitting. For larger models it is set to 0.1
    # Dropout rate of 10% means that 10% of the weights are dropped during training, 90% are used to train
    # the network. There is need to drop weights so that model is forced to learn more distributed patterns
    # It forces multipaths to learn the same features(better regularization).
    # It temporarily shrinks the model per forward pass.(Avoids memorizing noise).
    # It creates ensemble training like effect by exponentially training the sub-netowrks. It improves validation accuracy
    self.register_buffer( 
        "mask", # Name: Accessible as self.mask
        torch.triu(torch.ones(context_length, context_length), diagonal = 1)
    )# This line creates a causal attention mask as a non-trainable buffer in transformer preventing 
    # future future tokends from attending to past tokens.
    # torch.ones(y, y) creates a matrix of ones of size y x y
    # torch.triu(..., diagonal=1): Sets elements on/below diagonal=1 to 0.
    # torch.triu(..., diagonal=0): Sets elements on/below diagonal=0 to 0.
    

  def forward(self, x):# x is the input to the multihead attention layer. Its context vector is to be returned.
    # Attention mechanism

    # x shape = [b, num_tokens, d_in]

    b, num_tokens, d_in = x.shape # Unpacking the shape of the input
    # Input shape: batch_size * seq_len * d_in
    # b = batch_size
    # num_tokens = seq_len
    # d_in = d_in -> embedding dimension of each token

    query = self.W_query(x) # Query vector
    # Matrix multiplication: x @ W_query 
    # Size of x : batch_size * seq_len * d_in 
    # Size of W_query : d_in * d_out 
    # Size of query : batch_size * seq_len * d_out 
    # d_out is the output of dimension per head
    # total dimension of each vector will be d_out * head_dim

    key = self.W_key(x) # Key vector
    # Matrix multiplication: x @ W_key 
    # Size of x : batch_size * seq_len * d_in 
    # Size of W_key : d_in * d_out 
    # Size of key : batch_size * seq_len * d_out 

    value = self.W_value(x)
    # Matrix multiplication: x @ W_value 
    # Size of x : batch_size * seq_len * d_in 
    # Size of W_value : d_in * d_out 
    # Size of value : batch_size * seq_len * d_out 
    key = key.view(b, num_tokens, self.num_heads, self.head_dim)
    # key before view: batch_size * seq_len * d_out -> contains all the heads concatenated
    # d_out = 512 (8 heads * 64 dimensions)
    # key after view: batch_size * seq_len * num_heads * head_dim
    # This is done for parallel attention from each of the 8 heads(Ex: 8 parallel attention modules)

    query = query.view(b, num_tokens, self.num_heads, self.head_dim)
    # query after view: batch_size * seq_len * num_heads * head_dim
    # This is done for parallel attention from each of the 8 heads(Ex: 8 parallel attention modules)
    value = value.view(b, num_tokens, self.num_heads, self.head_dim)
    # value after view: batch_size * seq_len * num_heads * head_dim
    # This is done for parallel attention from each of the 8 heads(Ex: 8 parallel attention modules)
    key = key.transpose(1, 2)
    # key after transpose: batch_size * num_heads * seq_len * head_dim
    value = value.transpose(1, 2)
    # value after transpose: batch_size * num_heads * seq_len * head_dim
    query = query.transpose(1, 2)
    # query after transpose: batch_size * num_heads * seq_len * head_dim
    attn_score = query @ key.transpose(2, 3)
    # attn_score after @: batch_size * num_heads * seq_len * seq_len
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    # The buffer that we registered earler as self.mask. It created a upper triangular if size seq_length * seq_length. With 1 on upper side.
    # self.mask.bool() -> Converts to boolean type. With 0 as False and all other values as True.
    # seq_len is the maximum number of tokens in the input sequence. num_tokens might be less than seq_len. So it just reshapes
    # the mask to the size of num_tokens * num_tokens
    attn_score.masked_fill_(mask_bool, -torch.inf)
    # attn_score.masked_fill_(mask_bool, -torch.inf) -> Replaces all the values in attn_score where mask_bool is True with -inf
    attn_weights = torch.softmax(attn_score / key.shape[-1] ** 0.5, dim = -1)
    # attn_weights after softmax: batch_size * num_heads * seq_len * seq_len
    # Softmax is applied along the last dimension (dim = -1) to ensure that the sum of the weights for each token is 1.
    # Without scaling:
    # As d_k increases, the dot products QK^⊤ tend to have larger magnitude.
    # Large magnitudes pushed into softmax cause it to saturate:
    # One position gets probability ≈ 1, others ≈ 0.
    # Gradients through softmax become very small (vanishing gradients), making training unstable or slow.
    # With scaling:
    # The effect of d_k is reduced, making the dot products more stable.
    # Training becomes more stable and faster.
    attn_weights = self.dropout(attn_weights)
    # attn_weights after dropout: batch_size * num_heads * seq_len * seq_len
    # Dropout is applied to the attention weights to prevent overfitting.
    context_vec = (attn_weights @ value).transpose(1, 2)
    # context_vec after @: batch_size * num_heads * seq_len * head_dim
    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
    # context_vec after view: batch_size * seq_len * d_out
    context_vec = self.out_proj(context_vec)
    # context_vec after out_proj: batch_size * seq_len * d_out
    return context_vec