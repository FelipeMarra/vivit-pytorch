#%%
import sys
import os

sys.path.append(os.path.abspath('../'))

from datasets.get_datasets import get_datasets, DatasetsEnum
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchaudio.io import StreamWriter

DATASET_PATH = '/home/felipe/Desktop/games_dataset'
#DATASET_PATH = '/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/kinetics-dataset/k400/arranged'
BATCH_SIZE = 1
N_FRAMES = 32
EMB_DIM = 512
CROP_SIZE = 224
# tublets w/ 16x16 spatial patches and 2 time steps
#              T,  H,  W
TUBLET_SIZE = (2, 16, 16)

#%%
# Transforms will occur as [T, C, H, W], 
# before chuncks are transposed to [C, T, H, W] if dataloader transpose=True
test_transforms = v2.Compose([
        v2.Resize((CROP_SIZE, CROP_SIZE)),
        v2.ToDtype(torch.uint8),
    ])

# Passing test_transforms for test_transforms because we wont used anyway
train_dataset, eval_dataset, test_dataset = get_datasets(DatasetsEnum.GAMES, DATASET_PATH, N_FRAMES, 
                                                        test_transforms, test_transforms, transpose=False)

test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)

batch = next(iter(test_loader))

videos:torch.Tensor = batch['views']
classes = batch['class']
paths = batch['path']

# Tensor[B, V, T, C, H, W], where V is the number of views
print("batch videos shape:", videos.shape)
B, V, T, C, H, W = videos.shape

# %%
videos = videos.squeeze() # Remove batch dim 

# Video config
frame_rate = 30
height = CROP_SIZE
width = CROP_SIZE

for v_idx in range(V):
    video = videos[v_idx]
    print(v_idx, video.shape)

    # Configure stream
    s = StreamWriter(dst=f"./view_{v_idx}.mp4")
    s.add_video_stream(frame_rate=frame_rate, height=height, width=width, format="rgb24")

    # Generate video chunk (3 seconds)
    time = int(frame_rate * 3)
    chunk = video[:time]

    # Write data
    with s.open():
        s.write_video_chunk(0, chunk)
# %%
