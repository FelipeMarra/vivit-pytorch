#%%
from kinetics import KineticsDataset
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2
from vivit.vivit import ViViT
from train_utils import train

import matplotlib.pyplot as plt

# Dataset params
KINETICS_PATH = '/home/felipe/Desktop/k400/videos'
N_CLASSES = 35

# Video Params
N_FRAMES = 32
CROP_SIZE = 224

# Transformer Params
EMB_DIM = 1024
N_HEADS = 16 # head_size = emb_dim // n_heads
N_BLOCKS = 12 # num of decoder blocks
# tublets w/ 16x16 spatial patches and 2 time steps
#              T,  H,  W
TUBLET_SIZE = (2, 16, 16)
TUBLET_T,  TUBLET_H,  TUBLET_W = TUBLET_SIZE
N_PATCHES = (N_FRAMES/TUBLET_T) * (CROP_SIZE/TUBLET_H) * (CROP_SIZE/TUBLET_W)

assert N_PATCHES.is_integer()
N_PATCHES = int(N_PATCHES)

# General Params
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-3
EVAL_EVERY = 250

#%% 
# Loaders
# Transforms will occur as [T, C, H, W], before chuncks are transposed to [C, T, H, W]
train_transform = v2.Compose([
        #TODO:Normalize?
        v2.RandomResizedCrop((CROP_SIZE, CROP_SIZE))
    ])

train_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'train', N_FRAMES, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'val', N_FRAMES, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'test', N_FRAMES, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#%% 
# Model
model = ViViT(N_CLASSES, N_PATCHES, TUBLET_SIZE, EMB_DIM, N_HEADS, N_BLOCKS)

#%% 
# Train
train_loss, eval_loss = train(model, train_loader, val_loader, 
                              epochs=EPOCHS, lr=LR, eval_every=EVAL_EVERY)

#%% 
# Plot losses
fig = plt.figure()
plt.plot(train_loss, label='train')
plt.plot(eval_loss, label='eval')
fig.savefig('plot.png', dpi=fig.dpi)

print(f"Final losses: train {train_loss[-1]:.4f}; eval {eval_loss[-1]:.4f}")

#%%
# Save model
torch.save(model.state_dict(), './model.pth')