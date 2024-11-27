#%% Imports
import torch
from char_level_loader import CharLoader
from v1_decoder import V1Model, train

torch.manual_seed(1337)

#%% Consts
PATH = '/home/felipe/Documents/Github/video-transformer/v1_char_level/data/shakespeare.txt'
BATCH_SIZE = 4
CONTEXT_SIZE = 8

#%% Load some Shakespeare
loader = CharLoader(PATH, BATCH_SIZE, CONTEXT_SIZE)

#%% Test batch
x, y = loader.get_batch()

print("batch x:",loader.decode(x))
print("batch y:",loader.decode(y))

#%% Test untrained model
v1_model = V1Model(loader.vocab_size, loader.vocab_size)

# Loss
logits, loss = v1_model(x, y)
print("logits.shape:", logits.shape)
print("loss:",loss.item())

print("idx2char 0:", loader.idx2char[0])

# Generation
prompt = torch.zeros((1, 1), dtype=torch.long)
generated = v1_model.generate(prompt, 100)
decoded = loader.decode(generated)
print(decoded[0])

#%%
loss = train(v1_model, loader, steps=1000)
print("Final loss:", loss)

# Generation
prompt = torch.zeros((1, 1), dtype=torch.long)
generated = v1_model.generate(prompt, 100)
decoded = loader.decode(generated)
print(decoded[0])
# %%
