import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from vivit.vivit import ViViT

def train(model:ViViT, train_loader:DataLoader, val_loader:DataLoader, epochs:int, gpu_id:int, lr=1e-3, eval_every=100, writer:SummaryWriter|None=None):
    model = model.train()
    model = model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    global_step = 0
    running_loss = 0
    mean_train_loss = 0
    mean_eval_loss = 0
    acc_sum = 0

    for e_idx in range(epochs):
        train_loader.sampler.set_epoch(e_idx)
        tqdm.write(f"GPU {gpu_id}, EPOCH {e_idx}, N BATCHES {len(train_loader)} BATCH SIZE {train_loader.batch_size}")
        for b_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train"):
            xb = batch['video'].to(gpu_id)
            yb = batch['class'].to(gpu_id)

            # Mixed precision forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(xb)
                loss:torch.Tensor = criterion(logits, yb)

            # Scaled backprop
            scaler.scale(loss).backward()
            # Note: Use scaler.unscale_(opt) if need to change the grads
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss = loss.detach().cpu().item()

            # Stats and eval
            running_loss += loss
            if writer != None:
                writer.add_scalar('Train/Loss', loss, global_step)
                writer.add_scalar('Train/Lr', optimizer.param_groups[0]['lr'], global_step) # https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851/5

            mod_eval = (b_idx+1) % eval_every
            if mod_eval == 0 or b_idx+1 == len(train_loader):
                # train acc & loss
                train_acc = acc_sum/eval_every if mod_eval == 0 else acc_sum/mod_eval
                mean_train_loss = running_loss/eval_every if mod_eval == 0 else running_loss/mod_eval

                # eval acc & loss
                eval_acc, mean_eval_loss = eval(model, val_loader, writer, global_step, gpu_id)

                if writer != None:
                    writer.add_scalar('Train/Acc', train_acc, global_step)

                tqdm.write(f"\n Iter {b_idx+1} | TRAIN Acc: {train_acc:.4f}; Loss:  {mean_train_loss:.4f} | EVAL Acc: {eval_acc:.4f}; Loss: {mean_eval_loss:.4f} \n")

                running_loss = 0

            global_step += 1

        #%%
        # Save model
        if gpu_id == 0:
            if(not os.path.isdir("./models")):
                os.mkdir("./models")

            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": criterion.state_dict(),
                "scaler": scaler.state_dict()
            }

            torch.save(checkpoint, f"./models/model_ep_{e_idx+1}_trl_{mean_train_loss:.4f}_evl_{mean_eval_loss:.4f}.pth")

@torch.no_grad()
def eval(model:ViViT, loader:DataLoader, writer:SummaryWriter|None, global_step:int, gpu_id:int):
    model = model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_sum = 0
    acc_sum = 0
    for _, batch in tqdm(enumerate(loader), total=len(loader), desc="Eval"):
        xb = batch['video'].to(gpu_id)
        yb = batch['class'].to(gpu_id)

        # Mixed precision forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(xb)
            loss = criterion(logits, yb)
            loss = loss.detach().cpu().item()

        loss_sum += loss

        # acc
        predict = torch.argmax(logits, dim=1)
        acc_sum += torch.sum(predict == yb).detach().cpu().item()

    model = model.train()

    n_examples = len(loader)*loader.batch_size
    acc = acc_sum/n_examples
    loss = loss_sum/n_examples

    if writer != None:
        writer.add_scalar('Eval/Acc', acc, global_step)
        writer.add_scalar('Eval/Loss', loss, global_step)

    return acc, loss

@torch.no_grad()
def test(model:ViViT, loader:DataLoader, gpu_id:int, checkpoint=None, writer:SummaryWriter|None=None):
    criterion = nn.CrossEntropyLoss()

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        criterion.load_state_dict(checkpoint["optimizer"])

    model = model.eval()

    loss_sum = 0
    acc_sum = 0
    for _, batch in tqdm(enumerate(loader), total=len(loader), desc="Test"):
        # Batch will be [B, V, C, T, H, W], where V is the number of views
        xb:torch.Tensor = batch['views'].to(gpu_id)
        B, V, C, T, H, W = xb.shape

        # We'll put views of the same video one after the other along the batch dim
        xb = xb.view(B*V, C, T, H, W)

        yb = batch['class'].to(gpu_id)

        # Mixed precision forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # logits will be [B*V, n_classes]
            logits:torch.Tensor = model(xb)
            # we'll separate the views and avg them
            logits = logits.view(B, V, model.n_classes)
            logits = logits.mean(1) # keepdims = False will remove the V dim

            loss:torch.Tensor = criterion(logits, yb)
            loss = loss.detach().cpu().item()

        # acc
        predict = torch.argmax(logits, dim=1)
        acc_sum += torch.sum(predict == yb).detach().cpu().item()

        loss_sum += loss

    model = model.train()

    n_examples = len(loader)*loader.batch_size
    acc = acc_sum/n_examples
    loss = loss_sum/n_examples

    if writer != None:
        writer.add_scalar('Test/Acc', acc)
        writer.add_scalar('Test/Loss', loss)

    return loss, acc