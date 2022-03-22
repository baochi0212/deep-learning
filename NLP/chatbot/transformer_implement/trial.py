import torch
import torch.nn as nn

# # 
# x = torch.rand(1,  3, 11, 11)
# mask = torch.rand(1, 1, 1, 11)
# x = x.masked_fill(mask!=0, -1000)
# print(mask)
# print(x)
# x = torch.rand(32, 11)
# mask = torch.rand(32, 11)
# print((x  * mask).shape)


# MASK = torch.rand(1, 1, 2, 2)
# x = torch.rand(1, 8, 2, 2)
# x = x.masked_fill(MASK==0, -100)
# print(x.shape)
#transformer
x = torch.rand(32, 11, 128)
y = torch.rand(32, 11, 128)
model = nn.Transformer(d_model=128, batch_first=True)
print(model(x, y)[0])