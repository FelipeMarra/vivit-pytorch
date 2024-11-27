import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class LinearMultiheadAtt(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads

        self.wq = nn.Parameter(d_model, d_model)
        self.wk = nn.Parameter(d_model, d_model)
        self.wv = nn.Parameter(d_model, d_model)

    def forward(self, x, mask=True):
        pass

class ParallelMultiheadAtt(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads

        self.wq = nn.Parameter(d_model, d_model)
        self.wk = nn.Parameter(d_model, d_model)
        self.wv = nn.Parameter(d_model, d_model)
        self.wo = nn.Parameter(d_model, d_model)

    def forward(self, x, mask=True):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

class V1Model(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        # x -> (B, S); S = seq len
        emb_tokens = self.embedding_table(x) # emb_tokens -> (B, S, E); E = embeding dims
        logits = self.lm_head(emb_tokens) # lm_head (E, V) @ emb_tokens -> (B, S, V); V = vocab size

        if targets == None:
            loss = None
        else:
            B,S,V = logits.shape
            logits = logits.view(B*S, V)
            targets = targets.view(B*S)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, generated, max_num_tokens):
        # generated --> (B, S); an array of indices in the current sequence
        for _ in range(max_num_tokens):
            # get predictions from model
            logits, _ = self(generated) # logits --> (B, S, V)
            # get last chars
            logits = logits[:, -1, :] # (B, V)
            # softmax on the vocab dimention
            probs = F.softmax(logits, dim=-1) # (B, V)
            # sample from probs distribution
            preds = torch.multinomial(probs, 1) # (B, 1)
            # append preds on generated
            generated = torch.cat((generated, preds), dim=1) # (B, S+1)

        return generated

def train(model:nn.Module, loader, batch_size=32, steps=100, avg_steps=10):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    running_loss = 0
    for i in range(steps):
        # sample a batch of data
        xb, yb = loader.get_batch('train')

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i+1) % avg_steps == 0:
            mean_loss = running_loss/avg_steps
            losses.append(mean_loss)
            running_loss = 0
        else:
            loss_item = loss.item()
            running_loss += loss_item
            print(loss_item)

    return losses