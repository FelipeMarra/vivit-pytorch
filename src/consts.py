from datasets.get_datasets import DatasetsEnum

# Dataset params
DATASET = DatasetsEnum.KINETICS400
DATASET_PATH = '/home/es119256/kinetics-dataset/k400/videos'
#DATASET_PATH = '/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/kinetics-dataset/k400/arranged'
N_CLASSES = 400
N_WORKERS = 4 # Number of dataloader workrs per GPU

# Video Params
N_FRAMES = 32
MIN_RESIZE = 256
CROP_SIZE = 224

# Transformer Params
EMB_DIM = 768 
N_HEADS = 12 # head_size = emb_dim // n_heads
N_BLOCKS = 12 # num of decoder blocks
# tublets w/ 16x16 spatial patches and 2 time steps
#              T,  H,  W
TUBLET_SIZE = (2, 16, 16)
TUBLET_T,  TUBLET_H,  TUBLET_W = TUBLET_SIZE
N_PATCHES = (N_FRAMES/TUBLET_T) * (CROP_SIZE/TUBLET_H) * (CROP_SIZE/TUBLET_W)

# General Params
SEED = 1234
BATCH_SIZE = 4
TEST_BATCH_SIZE = 1 # Test with 4 views, so loades 4 videos for each BATCH_SIZE videos
EPOCHS = 30
LR = 1e-5
EVAL_EVERY = 1000