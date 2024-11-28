#%% Imports
import torch
from char_level_loader import CharLoader
from v1_decoder import V1Model, train
import matplotlib.pyplot as plt 

torch.manual_seed(1337)

#%% Consts
PATH = '/home/felipe/Documents/Github/video-transformer/v1_char_level/data/shakespeare.txt'
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
v1_model = V1Model(loader.vocab_size, CONTEXT_SIZE, EMBEDDING_DIM)

# Loss
logits, loss = v1_model(x, y)
print("logits.shape:", logits.shape)
print("loss:",loss.item())

print("idx2char 0:", loader.idx2char[0])

# Generation
def gen_test():
    prompt = torch.zeros((1, 1), dtype=torch.long)
    generated = v1_model.generate(prompt, 100)
    decoded = loader.decode(generated)
    print(decoded[0])

gen_test()

#%% Train and check results
losses = train(v1_model, loader, steps=1000)
plt.plot(range(len(losses)), losses)
print("Final loss:", losses[-1])

# Generation
gen_test()