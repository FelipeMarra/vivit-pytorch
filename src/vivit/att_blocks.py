import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

class MultiHeadAtt(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()

        # assert emb_dim % n_heads == 0 torch implementation will verify this already
        self.head_size = emb_dim // n_heads
        self.n_heads = n_heads

        # TODO no att dropout? 
        # No dropout at all? https://github.com/google-research/scenic/blob/97d6ac5b65040621f266b0da3bf05066baa664f3/scenic/projects/vivit/configs/kinetics400/vivit_base_k400.py#L101
        self.multihead_att = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)

        self.avg_att_weights = None

    def forward(self, x:torch.Tensor):
        # force attention implementation
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            # x is (B, S, E) & avg_att_weights is (B, S, E)
            x, self.avg_att_weights = self.multihead_att(query=x, key=x, value=x)

        return x