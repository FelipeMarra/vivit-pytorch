import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import islice

def train(model:nn.Module, writer:SummaryWriter, train_loader:DataLoader, val_loader:DataLoader, epochs:int, lr=1e-3, eval_every=100):
    model = model.train()
    model = model.cuda()

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss().cuda()
    scaler = torch.amp.GradScaler()

    global_step = 0
    running_loss = 0
    mean_train_loss = 0
    mean_eval_loss = 0

    for _ in range(epochs):
        for b_idx, batch in tqdm(islice(enumerate(train_loader),200), total=len(train_loader), desc="Train"):
            xb = batch['video'].cuda()
            yb = batch['class'].cuda()

            # Mixed precision forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(xb)
                loss = criterion(logits, yb)

            # Scaled backprop
            scaler.scale(loss).backward()
            # Note: Use scaler.unscale_(opt) if need to change the grads
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss = loss.detach().cpu().item()

            # Stats and eval
            running_loss += loss
            writer.add_scalar('Train/Loss', loss, global_step)
            writer.add_scalar('Train/Lr', optimizer.param_groups[0]['lr'], global_step) # https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851/5?u=felipe_marra

            steps_till_eval = (b_idx+1) % eval_every
            if steps_till_eval == 0 or b_idx+1 == len(train_loader):
                mean_train_loss = running_loss/(eval_every - steps_till_eval)
                mean_eval_loss = eval(model, val_loader, writer, global_step)

                tqdm.write(f"\n iter {b_idx+1} | train loss: {mean_train_loss:.4f}, eval loss: {mean_eval_loss:.4f} \n")

                running_loss = 0

            global_step += 1

        #%%
        # Save model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": criterion.state_dict(),
            "scaler": scaler.state_dict()
        }

        torch.save(checkpoint, f"./model_t_{mean_train_loss:.4f}_e_{mean_eval_loss:.4f}.pth")

@torch.no_grad()
def eval(model:nn.Module, loader:DataLoader, writer:SummaryWriter, global_step:int):
    model = model.eval()
    criterion = nn.CrossEntropyLoss().cuda()

    losses = torch.zeros(len(loader))
    for b_idx, batch in tqdm(islice(enumerate(loader),100), total=len(loader), desc="Eval"):
        xb = batch['video'].cuda()
        yb = batch['class'].cuda()

        # Mixed precision forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(xb)
            loss = criterion(logits, yb)
            loss = loss.detach().cpu().item()

        losses[b_idx] = loss

    model = model.train()

    loss = losses.mean().item()
    writer.add_scalar('Eval/Loss', loss, global_step)

    return loss

@torch.no_grad()
def test(model:nn.Module, loader:DataLoader, writer:SummaryWriter, checkpoint=None):
    criterion = nn.CrossEntropyLoss().cuda()

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        criterion.load_state_dict(checkpoint["optimizer"])

    model = model.eval()
    model = model.cuda()

    losses = torch.zeros(len(loader))
    acc_sum = 0
    for b_idx, batch in tqdm(islice(enumerate(loader),100), total=len(loader), desc="Test"):
        xb = batch['video'].cuda()
        yb = batch['class'].cuda()

        # Mixed precision forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(xb)
            loss = criterion(logits, yb)
            loss = loss.detach().cpu().item()

        # acc
        probs = F.softmax(logits, dim=1)
        predict = torch.argmax(probs, dim=1)
        acc_sum += torch.sum(predict == yb).detach().item()

        losses[b_idx] = loss

    model = model.train()

    n_examples = len(loader)*loader.batch_size
    acc = acc_sum/n_examples
    loss = losses.mean()

    writer.add_scalar('Test/Acc', acc)
    writer.add_scalar('Test/Loss', loss)

    return loss, acc