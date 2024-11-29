#%%
import torch

x = torch.randint(0, 10 , (4, 8, 4)).float()

B, S, C = x.shape

weights = torch.zeros(S, S)

mask = torch.tril(torch.ones(S, S))
print(mask)

weights = weights.masked_fill(mask == 0, float('-inf'))
print(weights)
print(weights.shape)

weights = torch.softmax(weights, dim=-1) #(S, S) (B, S, C)
print(weights)

print(x[0])

x = weights @ x
x[0]
# %%