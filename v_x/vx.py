#%%
from kinetics import KineticsDataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision.transforms import v2

KINETICS_PATH = '/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/kinetics-dataset/k400/arranged/'
N_FRAMES = 32
EMB_DIM = 512
CROP_SIZE = 224

#%%
# Transforms will occur as [T, C, H, W], before chuncks are transposed to [C, T, H, W]
train_transform = v2.Compose([
        #v2.Normalize((1,), (0,), inplace=True),
        v2.RandomCrop((CROP_SIZE, CROP_SIZE))
    ])

train_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'train', N_FRAMES, train_transform), batch_size=8, shuffle=True, num_workers=2)

batch = next(iter(train_loader))

videos = batch['video']
classes = batch['class']
paths = batch['path']

# Tensor[B, C, T, H, W]
print("batch videos shape:", videos.shape)
B, C, T, H, W = videos.shape

# %%
# tublets w/ 16x16 spatial patches and 2 time steps
#              T,  H,  W
tublet_size = (2, 16, 16)

tokenizer = nn.Conv3d(
        in_channels=3,
        out_channels=EMB_DIM,
        kernel_size=tublet_size,
        stride=tublet_size
    )
print('tokenizer weights:', tokenizer.weight.shape)

#%%
tokens:torch.Tensor = tokenizer(videos)
print(tokens.shape)

# %%
# Put embeds in function of the number of patches
tokens = tokens.view(B, -1, EMB_DIM)
print(tokens.shape)

# %%
print(paths)

for video_class in classes:
    video_class = video_class.item()
    print(video_class, train_loader.dataset.idx2class[video_class])
# %%
