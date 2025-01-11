#%%
import sys
import os
sys.path.append(os.path.abspath('../'))

from kinetics import KineticsDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import custom_transforms as cut

# Dataset params
KINETICS_PATH = '/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/kinetics-dataset/k400/arranged'
#KINETICS_PATH = '/root/kinetics-dataset/k400/videos'

# Video Params
N_FRAMES = 32
MIN_RESIZE = 256
CROP_SIZE = 224

# General Params
BATCH_SIZE = 1

#%% 
# Loaders
# Transforms will occur as [T, C, H, W], before chuncks are transposed to [C, T, H, W]
train_transform = v2.Compose([
        cut.ResizeSmallest(MIN_RESIZE),
        v2.RandomCrop((CROP_SIZE, CROP_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Transpose messes with the channels order, so for testing purposes it will be deactivated
train_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'train', N_FRAMES, train_transform, transpose=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

#%%
batch = next(iter(train_loader))

video = batch['video']

path = batch['path']
print(f"Original video path {path}")
print(f"Transformed video {video.dtype} {video.shape}")

#%%
# Check some frames
frames_idx = [0, 15, 31]

for frame_idx in frames_idx:
    frame:torch.Tensor = video[0, frames_idx, :, :]
    print(f"Frame {frame_idx+1}: shape {frame.shape}")
    print(frame)
    print()