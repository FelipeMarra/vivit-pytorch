import torch.nn as nn
from vivit.att_blocks import MultiHeadAtt

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, 4 * emb_dim)
        self.proj = nn.Linear(4 * emb_dim, emb_dim) # projection to the residual connetion
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = nn.ReLU()(self.linear(x))
        x = self.proj(x)
        return self.dropout(x)

class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()

        assert emb_dim % n_heads == 0
        head_size = emb_dim // n_heads

        self.norm1 = nn.LayerNorm(emb_dim)
        self.multi_head_att = MultiHeadAtt(emb_dim, head_size, n_heads)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)

    def forward(self, x):
        # pre-norm, multihead and add
        x = self.norm1(x)
        x = x + self.multi_head_att(x)

        # pre-norm, feed forward, & add
        x = self.norm2(x)
        x = x + self.ff(x)

        return x
