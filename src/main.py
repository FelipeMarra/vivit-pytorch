#%% 
# Imports
from datetime import datetime
import random
import multiprocessing as mp
import torch
import numpy as np
from consts import *
from kinetics import KineticsDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import custom_transforms as cut
from vivit.vivit import ViViT
from train_utils import train, test
from torch.utils.tensorboard import SummaryWriter

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

n_workers = mp.cpu_count() # n_workers == num of threads
train_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'train', N_FRAMES, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=n_workers)
val_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'val', N_FRAMES, test_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=n_workers)
test_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'test', N_FRAMES, test_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=n_workers)

#%%
# Tensorboard writer
now = datetime.now().strftime('%m/%d/%Y_%H:%M:%S')
writer = SummaryWriter(f'vivit_{now}')

#%% 
# Model
model = ViViT(N_CLASSES, N_PATCHES, TUBLET_SIZE, EMB_DIM, N_HEADS, N_BLOCKS)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# TODO not working: writer.add_graph(model.cuda(), next(iter(train_loader))['video'].cuda())

#%% 
# Train
train(model, writer, train_loader, val_loader, epochs=EPOCHS, lr=LR, eval_every=EVAL_EVERY)

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

loss, acc = test(model, test_loader, writer)
writer.close()
print(f"Test Loss {loss}, Acc {acc}")