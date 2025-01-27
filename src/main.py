#%% 
# Imports
import os
from datetime import datetime
import random
import torch
import numpy as np
import consts as c
from datasets.get_datasets import get_datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from vivit.vivit import ViViT
from train_utils import train, test
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as torch_mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

def main(rank:int, world_size:int):
    #%% 
    # Parameters asserts
    assert c.EMB_DIM % c.N_HEADS == 0, "EMB_DIM should be divisible by N_HEAD"
    assert c.N_PATCHES.is_integer(), "N_PATCHES should be an integer"
    N_PATCHES = int(c.N_PATCHES)

    #%%
    # Set seed
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    torch.cuda.manual_seed_all(c.SEED)

    #%%
    # Multi-GPU stuff
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    #%% 
    # Loaders
    # Transforms will occur as [T, C, H, W], before chuncks are transposed to [C, T, H, W]
    # RandomResizedCrop was used instead of scale jitter + random crop. DMVR scale jitter implementation is different from the torch one 
    # RandomResizedCrop value from  https://github.com/KSonPham/ViVit-a-Pytorch-implementation/blob/14aaab46a6301a1a786a6372f686d75b6ae7fac5/utils/data_utils.py#L54
    # RandomHorizontalFlip and RandomApply (for color jitter) values are from table 7 in the ViViT paper
    # ColorJitter values are same as DMVR default values: https://github.com/google-deepmind/dmvr/blob/77ccedaa084d29239eaeafddb0b2e83843b613a1/dmvr/processors.py#L602
    train_transforms = v2.Compose([
            v2.RandomResizedCrop((c.CROP_SIZE, c.CROP_SIZE), (0.05, 1)),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomApply([
                v2.ColorJitter(brightness=0.125, contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0.2)
            ], 
            0.8),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    test_transforms = v2.Compose([
            v2.Resize((c.CROP_SIZE, c.CROP_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset, eval_dataset, test_dataset = get_datasets(c.DATASET, c.DATASET_PATH, c.N_FRAMES, train_transforms, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, 
                            shuffle=False, num_workers=c.N_WORKERS, pin_memory=True,
                            sampler=DistributedSampler(train_dataset, shuffle=True))

    val_loader = DataLoader(eval_dataset, batch_size=c.BATCH_SIZE, 
                            shuffle=False, num_workers=c.N_WORKERS, pin_memory=True,
                            sampler=DistributedSampler(eval_dataset, shuffle=False))

    test_loader = DataLoader(test_dataset, batch_size=c.TEST_BATCH_SIZE, 
                            shuffle=False, num_workers=c.N_WORKERS, pin_memory=True,
                            sampler=DistributedSampler(test_dataset, shuffle=False))

    #%%
    # Tensorboard writer for GPU 0
    writer = None
    if rank == 0:
        now = datetime.now().strftime('%m/%d/%Y_%H:%M:%S')
        writer = SummaryWriter(f'vivit_{now}')

    #%% 
    # Model
    model = ViViT(c.N_CLASSES, N_PATCHES, c.TUBLET_SIZE, c.EMB_DIM, c.N_HEADS, c.N_BLOCKS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    # TODO not working: writer.add_graph(model.cuda(), next(iter(train_loader))['video'].cuda())

    #%% 
    # Train
    train(model, train_loader, val_loader, c.EPOCHS, rank, optim_enum=c.OPTIM, lr=c.LR, eval_every=c.EVAL_EVERY, writer=writer)

    #%%
    # Test
    #model = ViViT(N_CLASSES, N_PATCHES, TUBLET_SIZE, EMB_DIM, N_HEADS, N_BLOCKS)
    #model = model
    # dev = torch.cuda.current_device()
    # checkpoint = torch.load("filename",
    #                         map_location = lambda storage, loc: storage.cuda(dev))

    # If resuming train needs to load scaler
    # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#saving-resuming
    # model.load_state_dict(checkpoint["model"])
    # criterion.load_state_dict(checkpoint["optimizer"])
    # scaler.load_state_dict(checkpoint["scaler"])

    loss, acc = test(model, test_loader, rank, writer=writer)
    print(f"Test Loss {loss}, Acc {acc}")

    if writer != None:
        writer.close()

    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch_mp.spawn(main, args=(world_size,), nprocs=world_size)