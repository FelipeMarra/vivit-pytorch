import torch
import torch.nn as nn
from torch.nn import functional as F

class AvgSelfAtt(nn.Module):
    def __init__(self):
        super().__init__()
    

    def forward(self, x):
        _, S, _ = x.shape

        weights = torch.zeros(S, S).cuda() # zeros will become 1s on the softmax exp, creating a uniform distribution
        mask = torch.tril(torch.ones(S, S).cuda()) # zeros if above the main diagonal and ones otherwise
        weights = weights.masked_fill(mask == 0, float('-inf')) # set upper triangular values to -inf
        weights = torch.softmax(weights, dim=-1) # create a uniform distribution in relation to the 1 positions for each row, -inf position become 0
        x = weights @ x # average x's row values up until the main diagonal. weights (S, S) @ x (B, S, E) become (B, S, E) 

        return x

class CharV3(nn.Module):
    def __init__(self, vocab_size, context_size, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.emb_dim = emb_dim

        self.tkn_emb_table = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb_table = nn.Embedding(context_size, emb_dim)
        self.avg_att = AvgSelfAtt()
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        # x -> (B, S); S = seq len
        B, S = x.shape

        # token and positiona embeddings
        tkn_emb = self.tkn_emb_table(x) # tkn_emb -> (B, S, E); E = embeding dims

        seq_indexes = torch.arange(S).cuda()
        pos_emb = self.pos_emb_table(seq_indexes) # pos_emb -> (S, E)
        x = tkn_emb + pos_emb # (B, S, E)

        x = self.avg_att(x) # (B, S, E)

        logits = self.linear(x) # x (B, S, E) @ linear (E, V) -> (B, S, V); V = vocab size

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

@torch.no_grad()
def eval(model:nn.Module, loader, eval_steps:int):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            x, y = loader.get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


def train(model:nn.Module, loader, steps=100, lr=1e-3, eval_every=100, eval_steps=100):
    model.train()
    model = model.cuda()

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loss = []
    eval_loss = []
    for i in range(steps):
        # sample a batch of data
        xb, yb = loader.get_batch('train')

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i+1) % eval_every == 0:
            losses = eval(model, loader, eval_steps)
            train_loss.append(losses['train'])
            eval_loss.append(losses['val'])
            print(f"iter {i+1} | train loss: {losses['train']:.4f}, eval loss: {losses['val']:.4f}")

    return train_loss, eval_loss