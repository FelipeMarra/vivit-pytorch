#%%
from kinetics import KineticsDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from vivit.vivit import ViViT
from train_utils import train, test

import matplotlib.pyplot as plt

# Dataset params
#KINETICS_PATH = '/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/kinetics-dataset/k400/arranged'
KINETICS_PATH = '/root/kinetics-dataset/k400/videos'
N_CLASSES = 400

# Video Params
N_FRAMES = 32
CROP_SIZE = 224

# Transformer Params
EMB_DIM = 768 
N_HEADS = 6 # head_size = emb_dim // n_heads
N_BLOCKS = 8 # num of decoder blocks
# tublets w/ 16x16 spatial patches and 2 time steps
#              T,  H,  W
TUBLET_SIZE = (2, 16, 16)
TUBLET_T,  TUBLET_H,  TUBLET_W = TUBLET_SIZE
N_PATCHES = (N_FRAMES/TUBLET_T) * (CROP_SIZE/TUBLET_H) * (CROP_SIZE/TUBLET_W)

assert N_PATCHES.is_integer()
N_PATCHES = int(N_PATCHES)

# General Params
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-3
EVAL_EVERY = 10000

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
#state_dict = torch.load('/home/felipe/Desktop/model_t_nan_e_nan_epoch_1.pth', weights_only=True)
#model.load_state_dict(state_dict)
loss, acc = test(model, test_loader)

print(f"Test Loss {loss}, Acc {acc}")