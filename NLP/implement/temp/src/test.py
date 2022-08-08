import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import random
from copy import deepcopy


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
print("os var", os.environ['dir'])

print("random", random.uniform(0, 1))
a = torch.rand(3)
print(a)
b = deepcopy(a)
b += 1 
print(a)

# a = torch.rand(32, 1, 10)
# b = torch.triu(torch.ones(2, 3, 10), diagonal=1)
# print("dot", torch.bmm(a, a.permute(0, 2, 1)).shape)
# print("convert to diagonal", b)

# # x = torch.rand(2, 3)
# # y = torch.zeros(2, 3)
# # print(x * y)
# # class Property:
# #     '''
# #     - setter will set the private a value to x
# #     - property call setter and getter to self.content
    
# #     '''
# #     def __init__(self, content):
# #         #need the setter for using this, so we can have only-get constraint
# #         self.content = content
# #     #private
# #     @property
# #     def content(self):
# #         return self._a

# #     @content.setter
# #     def content(self, x):
# #         self._a = x

# # #test @property
# # obj = Property(3)
# # print("property", obj.content)


# class Property:
#     '''
#     - normal getter and setter for private _x, not directly use it
    
#     '''
#     def __init__(self, x):
#         #need the setter for using this, so we can have only-get constraint
#         self.set_x(x)
#     #private
#     def get_x(self):
#         return self._x
#     def set_x(self, x):
#         self._x = x
#     # @property
#     # def content(self):
#     #     return self._a

#     # @content.setter
#     # def content(self, x):
#     #     self._a = x

# #test @property
# obj = Property(3)
# print("property", obj.get_x()) #wanna directly use x, or .... (not _x) -> for betterment of refactoring

# class Property:
#     '''
#     -use x for access _x, x = property(get, set), still constrain and hiding data
    
#     '''
#     def __init__(self, x):
#         #need the setter for using this, so we can have only-get constraint
#         # self.x = x
#         #if don't wanna set _x, fix it with initial value and internal processing (only in-class operation)
#         self._x = 0 
#     #private
#     @property
#     def x(self):
#         return self._x #hide this

#     # @x.setter
#     # def x(self, value):
#     #     self._x = value

# #test @property
# obj = Property(3)
# print("property", obj.x)

# #mask fill
# a = torch.ones(5, 5)
# a = a.masked_fill(a==1, -1000)
# print(a)

# nn_seq = nn.Sequential(nn.Linear(3, 4), nn.Linear(1, 2))
# for module in nn_seq:
#     print(module)


# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# special_tokens_dict = {'additional_special_tokens': ['<SOS>','<EOS>']}
# # tokenizer.add_special_tokens(special_tokens_dict)
# # print("text", len(tokenizer("<SOS> what is that < EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>")['input_ids']))
# print("tokenizer", tokenizer.convert_ids_to_tokens(102))