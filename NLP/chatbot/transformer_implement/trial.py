import torch

x = torch.rand([2,5])
x[0, 0] = -1
y = torch.zeros_like(x) + torch.tensor([-100])
a = torch.where(x > 0, x, y)
print(a)

x = torch.tensor([1, 2, 3])
x = x.repeat(4, 2)
print(x.shape)