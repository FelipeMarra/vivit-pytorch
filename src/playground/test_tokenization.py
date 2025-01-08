#%%
import sys
import os

sys.path.append(os.path.abspath('../'))

from kinetics import KineticsDataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision.transforms import v2

KINETICS_PATH = '/home/felipe/Desktop/k400/videos'
BATCH_SIZE = 8
N_FRAMES = 32
EMB_DIM = 512
CROP_SIZE = 224
# tublets w/ 16x16 spatial patches and 2 time steps
#              T,  H,  W
TUBLET_SIZE = (2, 16, 16)

#%%
# Transforms will occur as [T, C, H, W], before chuncks are transposed to [C, T, H, W]
train_transform = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224))
    ])

train_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'train', N_FRAMES, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
eval_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'val', N_FRAMES, train_transform), batch_size=BATCH_SIZE, num_workers=2)
test_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'test', N_FRAMES, train_transform), batch_size=BATCH_SIZE, num_workers=2)

batch = next(iter(train_loader))

videos = batch['video']
classes = batch['class']
paths = batch['path']

# Tensor[B, C, T, H, W]
print("batch videos shape:", videos.shape)
B, C, T, H, W = videos.shape

# %%

tokenizer = nn.Conv3d(
        kernel_size=TUBLET_SIZE,
        stride=TUBLET_SIZE,
        in_channels=3,
        out_channels=EMB_DIM
    )
print('tokenizer weights:', tokenizer.weight.shape)

#%%
tokens:torch.Tensor = tokenizer(videos)
print(tokens.shape)

# %%
# Put embeds in function of the number of patches
tokens = tokens.view(B, -1, EMB_DIM)
print(tokens.shape)

#%%
T,  H,  W = TUBLET_SIZE
n_pathces = (N_FRAMES/T) * (CROP_SIZE/H) * (CROP_SIZE/W)
n_pathces

# %%
print(paths)

for video_class in classes:
    video_class = video_class.item()
    print(video_class, train_loader.dataset.idx2class[video_class])
# %%
