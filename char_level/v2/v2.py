#%% Imports
import torch
from char_level_loader import CharLoader
from v2_decoder import CharV2, train
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

setup_seed(42)

#%% Consts
PATH = '../data/shakespeare.txt'
BATCH_SIZE = 4
CONTEXT_SIZE = 8
EMBEDDING_DIM = 10

#%% Load some Shakespeare
loader = CharLoader(PATH, BATCH_SIZE, CONTEXT_SIZE)

#%% Test batch
x, y = loader.get_batch()

print("batch x:",loader.decode(x))
print("batch y:",loader.decode(y))

#%% Test untrained model
model = CharV2(loader.vocab_size, CONTEXT_SIZE, EMBEDDING_DIM)

# Loss
logits, loss = model(x, y)
print("logits.shape:", logits.shape)
print("loss:",loss.item())

print("idx2char 0:", loader.idx2char[0]) # \n

# Generation
def gen_test():
    prompt = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(prompt, 100)
    decoded = loader.decode(generated)
    print(decoded[0])

gen_test()

#%% Train and check results
losses = train(model, loader, steps=1500)
plt.plot(range(len(losses)), losses)
print("Final loss:", losses[-1])

# Generation
gen_test()