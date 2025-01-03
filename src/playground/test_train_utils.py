#%%
import torch
import torch.nn as nn
from torch.nn import functional as F

#%%
# Just checking one thing and I dont know why it doesnt really matter how hard you try just keep that in mind
criterion = nn.CrossEntropyLoss()

logits = torch.zeros((1,10))
yb = torch.tensor([5])

logits[0][yb] = 1

print(logits)
print(yb)

loss = criterion(logits, yb)
print(loss.item())

#%%
# Checking acc stuff
logits = torch.zeros((2,10))
yb = torch.tensor([5, 5])

logits[:, yb] = 1

print(logits)
print(yb)

probs = F.softmax(logits, dim=1)
print(probs)

predict = torch.argmax(probs, dim=1)
print(predict)

acc = torch.sum(predict == yb) / yb.shape[0]

print(f"Acc: {acc.item()*100}%")
# %%
