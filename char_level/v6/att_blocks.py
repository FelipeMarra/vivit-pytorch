import torch
import torch.nn as nn
from torch.nn import functional as F

class AttHead(nn.Module):
    def __init__(self, emd_dim, context_size, head_size):
        super().__init__()
        self.wq = nn.Linear(emd_dim, head_size, bias=False)
        self.wk = nn.Linear(emd_dim, head_size, bias=False)
        self.wv = nn.Linear(emd_dim, head_size, bias=False)
        self.dropout = nn.Dropout(0.2)

        self.register_buffer('mask', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        _, S, E = x.shape # (B, S, E)

        q:torch.Tensor = self.wq(x) # x (B, S, E) @ wq (E, E) = q (B, S, E)
        k:torch.Tensor = self.wk(x) # x (B, S, E) @ wk (E, E) = k (B, S, E)
        v:torch.Tensor = self.wv(x) # x (B, S, E) @ wv (E, E) = v (B, S, E)

        att = q @ k.transpose(-2, -1) # Words similarity q (B, S, E) @ k (B, E, S) = x (B, S, S)
        att:torch.Tensor = att * E**-0.5 # scale
        att = att.masked_fill(self.mask[:S, :S] == 0, float('-inf')) # apply mask
        att = F.softmax(att, dim=-1)  # Final attention weights (B, S, S)
        att = self.dropout(att)

        x = att @ v # Weighted average x (B, S, S) @ v (B, S, E) = x (B, S, E)

        return x

class MultiHeadAtt(nn.Module):
    def __init__(self, emb_dim, context_size, head_size, n_heads):
        super().__init__()

        self.heads = nn.ModuleList([AttHead(emb_dim, context_size, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim) # projection to the residual connetion
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # cat in the E dimention. if head size is E/2 and n_heads is 2, out will still be (B, S, E) 
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)