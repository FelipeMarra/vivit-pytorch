#%%
import sys
import os
sys.path.append(os.path.abspath('../'))

from datasets.datasets import get_datasets, DatasetsEnum
from torchaudio.io import StreamWriter
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# Dataset params
DATASET = DatasetsEnum.GAMES
DATASET_PATH = '/home/felipe/Desktop/database/5. Database/nintendo-snes-spc'
#DATASET_PATH = '/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/kinetics-dataset/k400/arranged'
#DATASET_PATH = '/root/kinetics-dataset/k400/videos'

# Video Params
N_FRAMES = 32
MIN_RESIZE = 256
CROP_SIZE = 224

# General Params
BATCH_SIZE = 1

#%% 
# Loaders
# Transforms will occur as [T, C, H, W], before chuncks are transposed to [C, T, H, W]
# RandomResizedCrop was used instead of scale jitter + random crop. DMVR scale jitter implementation is different from the torch one 
# RandomHorizontalFlip, RandomApply for color jitter values are from ViViT paper table 7
# ColorJitter values are same as DMVR default values: https://github.com/google-deepmind/dmvr/blob/77ccedaa084d29239eaeafddb0b2e83843b613a1/dmvr/processors.py#L602
train_transform = v2.Compose([
        v2.RandomResizedCrop((CROP_SIZE, CROP_SIZE), (0.05, 1)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomApply([
            v2.ColorJitter(brightness=0.125, contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0.2)
        ], 
        1.0),
    ])

train_dataset, eval_dataset, test_dataset = get_datasets(DATASET, DATASET_PATH, N_FRAMES, train_transform, None, transpose=False)

# Transpose messes with the channels order, so for testing purposes it will be deactivated
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

#%%
batch = next(iter(train_loader))

video = batch['video']
video_class = batch['class']

path = batch['path']
print(video)
print(f"Original video path {path} \n video class {video_class}")
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
