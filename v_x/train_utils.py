import torch
import torch.nn as nn
from tqdm import tqdm

@torch.no_grad()
def eval(model:nn.Module, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()

    losses = torch.zeros(len(loader))
    for b_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Eval"):
        xb = batch['video'].cuda()
        yb = batch['class'].cuda()
        #paths = batch['path']

        logits = model(xb)
        loss = criterion(logits, yb)
        losses[b_idx] = loss.item()

    model.train()

    return losses.mean()


def train(model:nn.Module, train_loader, val_loader, epochs, lr=1e-3, eval_every=100):
    model.train()
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
            tqdm.write(f"Y batch {yb}")
            #paths = batch['path']

            # evaluate the loss
            y_hat = model(xb)
            loss = criterion(y_hat, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (b_idx+1) % eval_every == 0:
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