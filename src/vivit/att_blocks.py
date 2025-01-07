from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

def attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, dropout:nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    head_size = query.size(-1)

    # Words similarity q (B, n_heads, S, head_size) @ k (B, n_heads, head_size, S) = x (B, n_heads, S, S)
    att = query @ key.transpose(-2, -1)
    # Scale by sqrt(head_size)
    att = att * head_size**-0.5
    # Final attention weights (B, n_heads, S, S)
    att = F.softmax(att, dim=-1)
    att = dropout(att)

    # Weighted average att (B, n_heads, S, S) @ v (B, n_heads, S, head_size) = x (B, n_heads, S, head_size)
    x = att @ value

    return x, att

class MultiHeadAtt(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()

        assert emb_dim % n_heads == 0
        self.head_size = emb_dim // n_heads
        self.n_heads = n_heads

        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        self.att = None
        self.proj = nn.Linear(emb_dim, emb_dim) # projection to the residual connetion
        self.dropout = nn.Dropout(0.1)

    def forward(self, x:torch.Tensor):
        B, S, E = x.shape

        query:torch.Tensor = self.wq(x)
        key:torch.Tensor = self.wk(x)
        value:torch.Tensor = self.wv(x)

        # q, k and v will be (B, n_heads, S, head_size)
        query = query.view(B, S, self.n_heads, self.head_size).transpose(1,2)
        key = key.view(B, S, self.n_heads, self.head_size).transpose(1,2)
        value = value.view(B, S, self.n_heads, self.head_size).transpose(1,2)

        # x is (B, n_heads, S, head_size)
        # att is (B, n_heads, S, S)
        x, self.att = attention(query, key, value, self.dropout)

        # transpose to (B, S, n_heads, head_size) 
        # and go back to (B, S, E)
        x = x.transpose(1, 2).contiguous().view(B, S, E)

        del query
        del key
        del value

        return self.wo(x)