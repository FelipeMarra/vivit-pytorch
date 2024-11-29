#%% Imports
import torch
from char_level_loader import CharLoader
from v4_decoder import CharV4, train
import matplotlib.pyplot as plt
import numpy as np
import random

#%% Seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#%% Consts
SEED = 42
PATH = '../data/shakespeare.txt'

STEPS = 10000 # num of training steps
BATCH_SIZE = 4
CONTEXT_SIZE = 8
EMBEDDING_DIM = 10
LR = 1e-3

EVAL_EVERY = 50 # eval every 50 train steps
EVAL_STEPS = 100

setup_seed(SEED)

#%% Load some Shakespeare
loader = CharLoader(PATH, BATCH_SIZE, CONTEXT_SIZE)

#%% Test batch
x, y = loader.get_batch()

print("batch x:",loader.decode(x))
print("batch y:",loader.decode(y))

#%% Test untrained model
model = CharV4(loader.vocab_size, CONTEXT_SIZE, EMBEDDING_DIM).cuda()

# Loss
logits, loss = model(x, y)
print("logits.shape:", logits.shape)
print("loss:",loss.item())

print("idx2char 0:", loader.idx2char[0]) # \n

# Generation
def gen_test():
    prompt = torch.zeros((1, 1), dtype=torch.long, device='cuda')
    generated = model.generate(prompt, 100)
    decoded = loader.decode(generated)
    print(decoded[0])

gen_test()

#%% Train
train_loss, eval_loss = train(model, loader, steps=STEPS, lr=LR, 
                              eval_every=EVAL_EVERY, eval_steps=EVAL_STEPS)

#%% Plot losses
plt.plot(train_loss, label='train')
plt.plot(eval_loss, label='eval')
print(f"Final losses: train {train_loss[-1]:.4f}; eval {eval_loss[-1]:.4f}")
leg = plt.legend()
plt.show()

#%% Generation
gen_test()