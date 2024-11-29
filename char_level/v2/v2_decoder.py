import torch
import torch.nn as nn
from torch.nn import functional as F

class CharV2(nn.Module):
    def __init__(self, vocab_size, context_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.emb_dim = emb_dim

        self.tkn_emb_table = nn.Embedding(vocab_size, emb_dim)
        # y = x @ A.T + b, where A is (vocab_size, emb_dim) and b in (1, vocab_size)
        # in practice it is x @ (emb_dim, vocab_size) + (1, vocab_size)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        # x -> (B, S); S = seq len
        B, S = x.shape

        # token and positiona embeddings
        tkn_emb = self.tkn_emb_table(x) # tkn_emb -> (B, S, E); E = embeding dims
        logits = self.linear(tkn_emb) # x (B, S, E) @ linear (E, V) -> (B, S, V); V = vocab size

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
            # get predictions from model with input being up to context_size
            _, S = generated.shape
            x = generated[:, -self.context_size:] if S > self.context_size else generated

            logits, _ = self(x) # logits --> (B, S, V)
            # get last chars
            logits = logits[:, -1, :] # (B, V)
            # softmax on the vocab dimention
            probs = F.softmax(logits, dim=-1) # (B, V)

            # sample from probs distribution
            preds = torch.multinomial(probs, 1) # (B, 1)
            # append preds on generated
            generated = torch.cat((generated, preds), dim=1) # (B, S+1)

        return generated

def train(model:nn.Module, loader, steps=100, avg_steps=10):
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