# Isolated file just for testig

#%%
import torch
from kinetics import KineticsDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from vivit.vivit import ViViT
from train_utils import train, test

# Dataset params
#KINETICS_PATH = '/home/felipe/Desktop/k400/videos/'
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
BATCH_SIZE = 64
LR = 1e-3

#%% 
# Loaders
# Transforms will occur as [T, C, H, W], before chuncks are transposed to [C, T, H, W]
train_transform = v2.Compose([
        #TODO:Normalize?
        v2.RandomResizedCrop((CROP_SIZE, CROP_SIZE))
    ])

test_loader = DataLoader(KineticsDataset(KINETICS_PATH, 'test', N_FRAMES, train_transform), batch_size=BATCH_SIZE, num_workers=4)

model = ViViT(N_CLASSES, N_PATCHES, TUBLET_SIZE, EMB_DIM, N_HEADS, N_BLOCKS)
state_dict = torch.load('/root/video-transformer/model_t_0.1865_e_6.1709.pth', weights_only=True)
model.load_state_dict(state_dict)
loss, acc = test(model, test_loader)

print(f"Test Loss {loss}, Acc {acc}")
