import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

def train(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, epochs:int, lr=1e-3, eval_every=100):
    model = model.train()
    model = model.cuda()

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().cuda()

    train_loss = []
    eval_loss = []
    running_loss = 0
    for e_idx in range(epochs):
        for b_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train"):
            xb = batch['video'].cuda()
            yb = batch['class'].cuda()

            # evaluate the loss
            y_hat = model(xb)
            loss = criterion(y_hat, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (b_idx+1) % eval_every == 0 or b_idx+1 == len(train_loader):
                mean_train_loss = running_loss/eval_every
                train_loss.append(mean_train_loss)
                running_loss = 0

                mean_eval_loss = eval(model, val_loader)
                eval_loss.append(mean_eval_loss)

                tqdm.write(f"iter {b_idx+1} | train loss: {mean_train_loss:.4f}, eval loss: {mean_eval_loss:.4f}")
            else:
                loss_item = loss.item()
                running_loss += loss_item

    return train_loss, eval_loss

@torch.no_grad()
def eval(model:nn.Module, loader:DataLoader):
    model = model.eval()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    losses = torch.zeros(len(loader))
    for b_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Eval"):
        xb = batch['video'].cuda()
        yb = batch['class'].cuda()

        logits = model(xb)
        loss = criterion(logits, yb)
        losses[b_idx] = loss.item()

    model = model.train()

    return losses.mean()

@torch.no_grad()
def test(model:nn.Module, loader:DataLoader):
    model = model.eval()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    losses = torch.zeros(len(loader))
    acc_sum = 0
    for b_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Test"):
        xb = batch['video'].cuda()
        yb = batch['class'].cuda()

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


def test_lr(model:nn.Module, loader):
    pass