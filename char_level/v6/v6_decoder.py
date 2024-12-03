import torch
import torch.nn as nn
from torch.nn import functional as F
from decoder_block import DecoderBlock

class CharV6(nn.Module):
    def __init__(self, vocab_size, context_size, emb_dim, n_heads, n_blocks):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.emb_dim = emb_dim

        self.tkn_emb_table = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb_table = nn.Embedding(context_size, emb_dim)
        self.blocks = nn.Sequential(*[DecoderBlock(emb_dim, context_size, n_heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(emb_dim, vocab_size)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        # x -> (B, S); S = seq len
        B, S = x.shape

        # token and positiona embeddings
        tkn_emb = self.tkn_emb_table(x) # tkn_emb -> (B, S, E); E = embeding dims

        seq_indexes = torch.arange(S).cuda()
        pos_emb = self.pos_emb_table(seq_indexes) # pos_emb -> (S, E)
        x = tkn_emb + pos_emb # (B, S, E)

        x = self.blocks(x) # (B, S, E)

        x = self.norm(x)
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
    model.eval()

    losses = torch.zeros(eval_steps)
    for k in range(eval_steps):
        x, y = loader.get_batch('val')
        _, loss = model(x, y)
        losses[k] = loss.item()

    model.train()

    return losses.mean()


def train(model:nn.Module, loader, steps=100, lr=1e-3, eval_every=100, eval_steps=100):
    model.train()
    model = model.cuda()

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loss = []
    eval_loss = []
    running_loss = 0
    for i in range(steps):
        # sample a batch of data
        xb, yb = loader.get_batch('train')

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i+1) % eval_every == 0:
            mean_train_loss = running_loss/eval_every
            train_loss.append(mean_train_loss)
            running_loss = 0

            mean_eval_loss = eval(model, loader, eval_steps)
            eval_loss.append(mean_eval_loss)

            print(f"iter {i+1} | train loss: {mean_train_loss:.4f}, eval loss: {mean_eval_loss:.4f}")
        else:
            loss_item = loss.item()
            running_loss += loss_item

    return train_loss, eval_loss