import torch
import torch.nn as nn
from vivit.encoder_block import EncoderBlock

class ViViT(nn.Module):
    def __init__(self, n_classes, n_patches, tublet_size, emb_dim, n_heads, n_blocks):
        super().__init__()
        self.n_patches = n_patches
        self.emb_dim = emb_dim
        self.n_classes = n_classes

        self.tokenizer = nn.Conv3d(kernel_size=tublet_size, stride=tublet_size, in_channels=3, out_channels=emb_dim)
        self.pos_emb_table = nn.Embedding(n_patches, emb_dim)
        self.encoder_blocks = nn.Sequential(*[EncoderBlock(emb_dim, n_heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, n_classes)

    def forward(self, x:torch.Tensor):
    #   B, C, T, H, W
        B, _, _, _, _ = x.shape

        # tokenize video and add positional embeddings
        tkn_emb:torch.Tensor = self.tokenizer(x)
        tkn_emb = tkn_emb.view(B, -1, self.emb_dim)

        seq_indexes = torch.arange(self.n_patches).cuda()
        pos_emb = self.pos_emb_table(seq_indexes) # pos_emb -> (P, emb_dim);  P = n patches
        x = tkn_emb + pos_emb # (B, S, E); E = embedding size

        # Append cls token
        cls = torch.zeros(B, 1, self.emb_dim, device=x.device, dtype=x.dtype)
        x = torch.cat((cls, x), dim=1)

        # Apply transformer encoder blocks
        x = self.encoder_blocks(x) # (B, P, E)

        # Get CLS tokens
        x = x[:, 0] # (B, 1, E)

        x = self.norm(x)
        logits = self.linear(x).squeeze(dim=1) # x (B, 1, E) @ linear (E, n_classes) -> (B, 1, n_classes)

        return logits