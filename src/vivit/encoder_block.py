import torch.nn as nn
from vivit.att_blocks import MultiHeadAtt

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, 4 * emb_dim) # they use 3072 wich is 4 * 768 (they embed dim)
        self.proj = nn.Linear(4 * emb_dim, emb_dim) # projection to the residual connetion
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = nn.GELU()(self.linear(x))
        # TODO: Dropout? https://github.com/google-research/scenic/blob/97d6ac5b65040621f266b0da3bf05066baa664f3/scenic/model_lib/layers/attention_layers.py#L399
        x = self.proj(x)
        return self.dropout(x)

class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.multi_head_att = MultiHeadAtt(emb_dim, n_heads)
        self.dropout = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)

    def forward(self, x):
        # pre-norm, multihead, add and dropout
        x = self.norm1(x)
        x = x + self.multi_head_att(x)
        x = self.dropout(x)

        # pre-norm, feed forward, & add
        x = self.norm2(x)
        x = x + self.ff(x)

        return x
