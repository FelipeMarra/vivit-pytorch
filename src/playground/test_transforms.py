#%%
import sys
import os
sys.path.append(os.path.abspath('../'))

from kinetics import KineticsDataset
import torch
from torchaudio.io import StreamWriter
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
        #TODO:Normalize?
        cut.ResizeSmallest(MIN_RESIZE),
        v2.RandomCrop((CROP_SIZE, CROP_SIZE))
    ])

# Transpose messes with the channels order, so for testing purposes it will be deactivated
train_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'train', N_FRAMES, train_transform, transpose=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

#%%
batch = next(iter(train_loader))

video = batch['video']

path = batch['path']
print(f"Original video path {path}")
print(f"Transformed video {video.dtype} {video.shape}")

video = video.squeeze() # Remove batch dim 

# Video config
frame_rate = 30
height = CROP_SIZE
width = CROP_SIZE

# Configure stream
s = StreamWriter(dst="./video.mp4")
s.add_video_stream(frame_rate=frame_rate, height=height, width=width, format="rgb24")

# Generate video chunk (3 seconds)
time = int(frame_rate * 3)
chunk = video[:time]

# Write data
with s.open():
    s.write_video_chunk(0, chunk)
# %%
