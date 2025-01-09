import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

def train(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, epochs:int, lr=1e-3, eval_every=100):
    model = model.train()
    model = model.cuda()

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss().cuda()
    scaler = torch.cuda.amp.GradScaler()

    train_loss = []
    eval_loss = []
    running_loss = 0
    mean_train_loss = 0
    mean_eval_loss = 0
    for e_idx in range(epochs):
        for b_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train"):
            xb = batch['video'].cuda()
            yb = batch['class'].cuda()

            # Mixed precision forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # evaluate the loss
                logits = model(xb)
                loss = criterion(logits, yb)

            # Scaled backprop
            scaler.scale(loss).backward()
            # Use scaler.unscale_(opt) if need to change the grads
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Stats and eval
            loss_item = loss.item()
            running_loss += loss_item

            if (b_idx+1) % eval_every == 0 or b_idx+1 == len(train_loader):
                mean_train_loss = running_loss/eval_every #TODO: This will fail if last batch is not full
                train_loss.append(mean_train_loss)

                mean_eval_loss = eval(model, val_loader)
                eval_loss.append(mean_eval_loss)

                tqdm.write(f"\n iter {b_idx+1} | train loss: {mean_train_loss:.4f}, eval loss: {mean_eval_loss:.4f} \n")

                running_loss = 0

        #%%
        # Save model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": criterion.state_dict(),
            "scaler": scaler.state_dict()
        }

        torch.save(checkpoint, f"./model_t_{mean_train_loss:.4f}_e_{mean_eval_loss:.4f}.pth")
        write_losses(f'./losses_e_{e_idx}.json', train_loss, eval_loss)

    return train_loss, eval_loss

@torch.no_grad()
def eval(model:nn.Module, loader:DataLoader):
    model = model.eval()
    criterion = nn.CrossEntropyLoss().cuda()

    losses = torch.zeros(len(loader))
    for b_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Eval"):
        xb = batch['video'].cuda()
        yb = batch['class'].cuda()

        # Mixed precision forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(xb)
            loss = criterion(logits, yb)

        losses[b_idx] = loss.item()

    model = model.train()

    return losses.mean().item()

@torch.no_grad()
def test(model:nn.Module, loader:DataLoader, checkpoint=None):
    criterion = nn.CrossEntropyLoss().cuda()

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        criterion.load_state_dict(checkpoint["optimizer"])

    model = model.eval()
    model = model.cuda()

    losses = torch.zeros(len(loader))
    acc_sum = 0
    for b_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Test"):
        xb = batch['video'].cuda()
        yb = batch['class'].cuda()

        # Mixed precision forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(xb)

            # acc
            probs = F.softmax(logits, dim=1)
            predict = torch.argmax(probs, dim=1)
            acc_sum += torch.sum(predict == yb).detach().item()

            # loss
            loss = criterion(logits, yb)

        losses[b_idx] = loss.item()

    model = model.train()

    n_examples = len(loader)*loader.batch_size
    return losses.mean(), acc_sum/n_examples


def write_losses(path:str, train_loss:list, eval_loss:list):
    losses_dic = {
        'train': str(train_loss),
        'eval': str(eval_loss)
    }

    with open(path, "w") as json_file: 
        json.dump(losses_dic, json_file, indent=4)