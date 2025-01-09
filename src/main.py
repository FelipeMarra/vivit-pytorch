#%% 
# Imports
import random
import torch
import numpy as np
from consts import *
from kinetics import KineticsDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import custom_transforms as cut
from vivit.vivit import ViViT
from train_utils import train, test
import matplotlib.pyplot as plt

#%% 
# Parameters asserts
assert EMB_DIM % N_HEADS == 0, "EMB_DIM should be divisible by N_HEAD"
assert N_PATCHES.is_integer(), "N_PATCHES should be an integer"
N_PATCHES = int(N_PATCHES)

#%%
# Set seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#%% 
# Loaders
# Transforms will occur as [T, C, H, W], before chuncks are transposed to [C, T, H, W]
train_transform = v2.Compose([
        cut.ResizeSmallest(MIN_RESIZE),
        v2.RandomCrop((CROP_SIZE, CROP_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        cut.ZeroCenterNorm()
    ])

test_transform = v2.Compose([
        v2.Resize((CROP_SIZE, CROP_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        cut.ZeroCenterNorm()
    ])

train_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'train', N_FRAMES, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'val', N_FRAMES, test_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'test', N_FRAMES, test_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

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