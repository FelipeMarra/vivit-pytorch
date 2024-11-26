#%% Imports
from char_level_loader import ShakeLoader

#%% Consts
PATH = '/home/felipe/Documents/Github/video-transformer/v1_char_level/data/shakespeare.txt'
BATCH_SIZE = 4
CONTEXT_SIZE = 8

#%% Get loader
loader = ShakeLoader(PATH, BATCH_SIZE, CONTEXT_SIZE)

#%% Test batch
x, y = loader.get_batch()

print("batch x:",loader.decode(x))
print("batch y:",loader.decode(y))