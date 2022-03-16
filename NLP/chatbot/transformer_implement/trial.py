import torch

y = torch.rand(2,4)
print(y)
idx = torch.tensor([1,2], dtype=torch.long).view(-1, 1)
print(torch.gather(y, 1, idx))