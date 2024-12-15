import torch
import torch.nn as nn
from vivit.encoder_block import EncoderBlock
from tqdm import tqdm

class ViViT(nn.Module):
    def __init__(self, n_classes, n_patches, tublet_size, emb_dim, n_heads, n_blocks):
        super().__init__()
        self.n_patches = n_patches
        self.emb_dim = emb_dim

        self.tokenizer = nn.Conv3d(kernel_size=tublet_size, stride=tublet_size, in_channels=3, out_channels=emb_dim)
        self.pos_emb_table = nn.Embedding(n_patches, emb_dim)
        self.blocks = nn.Sequential(*[EncoderBlock(emb_dim, n_heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(emb_dim)   
        self.linear = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        # x -> (B, C, T, H, W)
        #print(x.shape)
        B, C, T, H, W = x.shape

        # tokenize video and add positional embeddings
        tkn_emb:torch.Tensor = self.tokenizer(x)
        tkn_emb = tkn_emb.view(B, -1, self.emb_dim)

        seq_indexes = torch.arange(self.n_patches).cuda()
        pos_emb = self.pos_emb_table(seq_indexes) # pos_emb -> (P, emb_dim);  P = n patches
        x = tkn_emb + pos_emb # (B, S, E); E = embedding size

        #tqdm.write(f"X shape before cls {x.shape}")
        # Append cls token
        cls = torch.zeros(B, 1, self.emb_dim, device='cuda', dtype=torch.float32)
        x = torch.cat((cls, x), dim=1)
        #tqdm.write(f"X shape afeter cls {x.shape}")
        #tqdm.write(f"CLS is in the first position {x[:, 0]}")

        # Apply transformer blocks
        x = self.blocks(x) # (B, P, E)

        # Get cls tokens
        x = x[:, 0] # (B, 1, E)
        #tqdm.write(f"CLS {cls.shape}")

        x = self.norm(x)
        logits = self.linear(x).squeeze() # x (B, 1, E) @ linear (E, n_classes) -> (B, 1, n_classes)
        #tqdm.write(f"Logits {logits.shape}")

        return logits
