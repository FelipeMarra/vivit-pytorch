#%%
import sys
import os
sys.path.append(os.path.abspath('../'))

from kinetics import KineticsDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

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
        v2.RandomResizedCrop((CROP_SIZE, CROP_SIZE), (0.05, 1)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Transpose messes with the channels order, so for testing purposes it will be deactivated
train_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'train', N_FRAMES, train_transform, transpose=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

#%%
batch = next(iter(train_loader))

video1 = batch['video']
path = batch['path']
print(f"Original video path {path}")
print(f"Transformed video {video1.dtype} {video1.shape}")

video2 = train_transform(torch.zeros((1, 32, 3, 600, 600), dtype=torch.uint8))
video3 = train_transform(torch.ones((1, 32, 3, 500, 500), dtype=torch.uint8) * 255)
video4 = train_transform(torch.ones((1, 32, 3, 1280, 720), dtype=torch.uint8) * 125)

videos = [video1, video2, video3, video4]

#%%
# Check some frames
frames_idx = [0, 15, 31]

for idx, video in enumerate(videos):
    print(f"VIDEO {idx}")
    for frame_idx in frames_idx:
        frame:torch.Tensor = video[0, frames_idx, :, :]
        print(f"Frame {frame_idx+1}: shape {frame.shape}, type {frame.dtype}")
        print(frame)
        print()