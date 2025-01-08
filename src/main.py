#%%
from kinetics import KineticsDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import custom_transforms as cut
from vivit.vivit import ViViT
from train_utils import train, test

import matplotlib.pyplot as plt

# Dataset params
KINETICS_PATH = '/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/kinetics-dataset/k400/arranged'
#KINETICS_PATH = '/root/kinetics-dataset/k400/videos'
N_CLASSES = 400

# Video Params
N_FRAMES = 32
MIN_RESIZE = 256
CROP_SIZE = 224

# Transformer Params
EMB_DIM = 768 
N_HEADS = 12 # head_size = emb_dim // n_heads
N_BLOCKS = 1 # num of decoder blocks
# tublets w/ 16x16 spatial patches and 2 time steps
#              T,  H,  W
TUBLET_SIZE = (2, 16, 16)
TUBLET_T,  TUBLET_H,  TUBLET_W = TUBLET_SIZE
N_PATCHES = (N_FRAMES/TUBLET_T) * (CROP_SIZE/TUBLET_H) * (CROP_SIZE/TUBLET_W)

assert N_PATCHES.is_integer()
N_PATCHES = int(N_PATCHES)

# General Params
BATCH_SIZE = 1
EPOCHS = 30
LR = 1e-3
EVAL_EVERY = 10000

#%% 
# Loaders
# Transforms will occur as [T, C, H, W], before chuncks are transposed to [C, T, H, W]
train_transform = v2.Compose([
        cut.ResizeSmallest(MIN_RESIZE),
        v2.RandomCrop((CROP_SIZE, CROP_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        cut.ZeroCenterNorm()
    ])

train_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'train', N_FRAMES, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'val', N_FRAMES, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'test', N_FRAMES, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#%% 
# Model
model = ViViT(N_CLASSES, N_PATCHES, TUBLET_SIZE, EMB_DIM, N_HEADS, N_BLOCKS)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")
#print(model)

#%% 
# Train
train_loss, eval_loss = train(model, train_loader, val_loader, 
                              epochs=EPOCHS, lr=LR, eval_every=EVAL_EVERY)

#%% 
# Plot losses
fig = plt.figure()
plt.plot(train_loss, label='train')
plt.plot(eval_loss, label='eval')
plt.legend(['train', 'eval']) 
plt.xlabel("every 40k examples")
plt.ylabel("loss")
fig.savefig('plot.png', dpi=fig.dpi)

print(f"Final losses: train {train_loss[-1]:.4f}; eval {eval_loss[-1]:.4f}")

#%%
# Test
#model = ViViT(N_CLASSES, N_PATCHES, TUBLET_SIZE, EMB_DIM, N_HEADS, N_BLOCKS)
#model = model
# dev = torch.cuda.current_device()
# checkpoint = torch.load("filename",
#                         map_location = lambda storage, loc: storage.cuda(dev))

# If resuming train needs to load scaler
# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#saving-resuming
# model.load_state_dict(checkpoint["model"])
# criterion.load_state_dict(checkpoint["optimizer"])
# scaler.load_state_dict(checkpoint["scaler"])

loss, acc = test(model, test_loader)

print(f"Test Loss {loss}, Acc {acc}")