import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

#positional encoding
class Positional_Encoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
#for create softmax from the masked value and the bool mask (because the masked value is a list, we need to convert to matrix with sparse values for calculating the softmax)
def masked_value(value, mask):
    
    #masked pos
    masked_pos = torch.where(mask != 0, 1, torch.zeros_like(mask))
    masked_pos = masked_pos.float()
    for i in range(masked_pos.shape[0]):
        for j in range(masked_pos.shape[1]):
            if masked_pos[i][j] == 1:
                masked_pos[i][j] = value[0]
                value.pop(0)
            else:
                masked_pos[i][j] = -1000
    return masked_pos

#attention score
def attention_score(q, k, v, valid_lens, max_len, type='dot'):
    #mask batch x q_num_step
    #q, k, v : batch x num_step x qkv features
    #q@k: batch x num_step x num_step
    if type == 'dot':
        # mask_value = torch.tensor([[1 if i < valid_lens[j] else 0 for i in range(max_len)] for j in range(valid_lens.shape[0])])
        # mask_bool = torch.tensor([[True if i < valid_lens[j] else False for i in range(max_len)] for j in range(valid_lens.shape[0])])
        # print('mask', mask_bool.shape)
        dot_product_batch = torch.bmm(q, k.permute(0, 2, 1))/q.shape[-1]**0.5
        # print('dot batch', dot_product_batch.shape)
        for i in range(dot_product_batch.shape[0]): #get the batch value 
            #get the dot product of each batch #shape q_step x k_step
            valid_len = valid_lens[i]
            dot_product = dot_product_batch[i]
            # print('dot product', dot_product.shape)
            #mask for this batch, take into account if index < valid_len
            mask_bool = torch.tensor([[True if j < valid_len else False for j in range(dot_product.shape[1])] for k in range(dot_product.shape[0])])
            mask_value = torch.tensor([[1 if j < valid_len else 0 for j in range(dot_product.shape[1])] for k in range(dot_product.shape[0])])
            # print('mask', mask_bool.shape)
            dot_product = dot_product.masked_select(mask_bool)
            masked_product = masked_value(list(dot_product), mask_value)
            # print('masked product', masked_product)
            softmax = F.softmax(masked_product, dim=-1).unsqueeze(0)
            # print('softmax', softmax)
            if i == 0:
                out_softmax = softmax 
            else:
                out_softmax = torch.concat([out_softmax, softmax], dim=0)
        
        return torch.bmm(out_softmax, v)
    






#multi-head attention
class Multi_head_attention(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        assert in_dim % num_heads == 0
        assert hidden_dim % num_heads == 0 
        #q, k, v batch x n_step x qkv features -> split into n_heads -> concat -> project to the original dim
        self.hidden_q = nn.Linear(in_dim//num_heads, hidden_dim//num_heads)
        self.hidden_k = nn.Linear(in_dim//num_heads, hidden_dim//num_heads)
        self.hidden_v = nn.Linear(in_dim//num_heads, hidden_dim//num_heads)
        self.out = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, q, k, v, valid_lens, max_len):
        weighted_values = [] #weighted attenton
        #split and project to the hidden dim, calculate the attention and concat the attention_weighted values
        split = q.shape[-1]//self.num_heads
        for i in range(self.num_heads):
            q_i = self.hidden_q(q[:, :, i*split:(i+1)*split])
            k_i = self.hidden_k(k[:, :, i*split:(i+1)*split])
            v_i = self.hidden_v(v[:, :, i*split:(i+1)*split])
            weighted_values.append(attention_score(q_i, k_i, v_i, valid_lens, max_len))
        return self.out(torch.cat(weighted_values, dim=-1))



#layer norm skip connection 
def AddNorm(x1, x2, dropout=0.1):
    #x1 is the attention
    layernorm = nn.LayerNorm(x1.shape[-1])
    dropout = nn.Dropout(dropout)
    return layernorm(dropout(x1) + x2)

#shallow NN
class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

        

if __name__ == "__main__":
    # #trial of functions and classes
    # #embedding and normal funcs
    # x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    # embedding = nn.Embedding(1000, 128)
    # ffw = FeedForward(128, 256, 128)
    # pe = Positional_Encoding(128)
    # x = embedding(x)
    # print((x + pe(x)).shape)
    # print(AddNorm(x, x).shape)
    # print(ffw(x).shape)
    #attention trial
    q = torch.rand([32, 9, 128])
    k = torch.rand([32, 12, 128])
    v = torch.rand([32, 12, 128])
    valid_lens = torch.tensor(np.random.randint(1, 9, 32)) #batch 
    print('batch valid lens', valid_lens.shape)
    print('Attention score', attention_score(q, k, v, valid_lens, max_len=12).shape)
    
    # #multi head
    # attention = Multi_head_attention(128, 256, 128)
    # print('multi head', attention(q, k, v, valid_lens, max_len=12).shape) #batch x nstep x feature 
    # print('layernorm', AddNorm(q, q).shape)
