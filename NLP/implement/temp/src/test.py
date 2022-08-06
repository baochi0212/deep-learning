import torch
import torch.nn as nn
a = torch.rand(32, 1, 10)
b = torch.triu(torch.ones(2, 3, 10), diagonal=1)
print("dot", torch.bmm(a, a.permute(0, 2, 1)).shape)
print("convert to diagonal", b)

# x = torch.rand(2, 3)
# y = torch.zeros(2, 3)
# print(x * y)

nn_seq = nn.Sequential(nn.Linear(3, 4), nn.Linear(1, 2))
for module in nn_seq:
    print(module)

