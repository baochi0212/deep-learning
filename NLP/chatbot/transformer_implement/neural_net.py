import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
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

def create_mask(input, target_input):
    #the encoder mask:
    input_mask = input!=0 #b x max
    input_mask = input.unsqueeze(1).unsqueeze(1)
    #the decoder input mask:
    target_mask = target_input!=0
    print('TARGET MASK', target_mask.shape)
    autoregressive_mask = torch.triu(torch.ones(target_input.shape[-1], target_input.shape[-1])).transpose(1, 0).type_as(target_mask) #max x max
    target_mask = target_mask.unsqueeze(1) & autoregressive_mask.unsqueeze(0) #batch x max x max 
    target_mask = target_mask.unsqueeze(1) 
    return input_mask, target_mask
#attention score
def attention_score(q, k, v, type='dot', mode='encoder'):
    # #mask batch x q_num_step
    # #q, k, v : batch x num_step x qkv features
    # #q@k: batch x num_step x num_step
    # if type == 'dot':
    #     # mask_value = torch.tensor([[1 if i < valid_lens[j] else 0 for i in range(max_len)] for j in range(valid_lens.shape[0])])
    #     # mask_bool = torch.tensor([[True if i < valid_lens[j] else False for i in range(max_len)] for j in range(valid_lens.shape[0])])
    #     # print('mask', mask_bool.shape)
    #     dot_product_batch = torch.bmm(q, k.permute(0, 2, 1))/q.shape[-1]**0.5
    #     # print('dot batch', dot_product_batch.shape)
    #     for i in range(dot_product_batch.shape[0]): #get the batch value 
    #         #get the dot product of each batch #shape q_step x k_step
    #         valid_len = valid_lens[i]
    #         dot_product = dot_product_batch[i]
    #         # print('dot product', dot_product.shape)
    #         #mask for this batch, take into account if index < valid_len
    #         mask_bool = torch.tensor([[True if j < valid_len else False for j in range(dot_product.shape[1])] for k in range(dot_product.shape[0])]).to(device)
    #         mask_value = torch.tensor([[1 if j < valid_len else 0 for j in range(dot_product.shape[1])] for k in range(dot_product.shape[0])]).to(device)
    #         # print('mask', mask_bool.shape)
    #         dot_product = dot_product.masked_select(mask_bool)
    #         masked_product = masked_value(list(dot_product), mask_value)
    #         # print('masked product', masked_product)
    #         softmax = F.softmax(masked_product, dim=-1).unsqueeze(0)
    #         # print('softmax', softmax)
    #         if i == 0:
    #             out_softmax = softmax 
    #         else:
    #             out_softmax = torch.concat([out_softmax, softmax], dim=0)
    if type  == 'dot':
        #q (m x f) @ k(n x f) -> weight (m x n) <attention of query m to key n -> need to mask the padded in n> @ v (n x f) -> queried value : m x f
        #mask 1 1 1 0 
        if mode == 'encoder':
            dot_product = torch.bmm(q, k.permute(0, 2, 1))/(q.shape[-1]**0.5)
            # mask = torch.zeros_like(dot_product).to(device) + torch.tensor([-min]).to(device)
            # enc_mask = torch.zeros([q.shape[1], k.shape[1]]) #m x n

            # print('mask', mask.shape)

            # dot_product = torch.where(dot_product > 0, dot_product, mask).to(device)
        elif mode == 'none':
            dot_product = torch.bmm(q, k.permute(0, 2, 1))/(q.shape[-1]**0.5)
        else:
            #decoder valid lens: torch.arange(1, num_steps(keys)) -> because the query can only attend the seen keys
            dot_product = torch.bmm(q, k.permute(0, 2, 1))/(q.shape[-1]**0.5) # m x n shape -> need n x n mask
            num_steps = k.shape[1] #key num step
            mask = torch.tensor([[True if pos < idx + 1 else False for pos in range(num_steps)] for idx in range(q.shape[1])]).repeat(dot_product.shape[0], 1, 1).float().to(device)

            # print('mask', mask)
            
            dot_product = dot_product.masked_fill(mask==0, -10000)
            # print('dot product', dot_product)
            # mask = torch.tensor([[1 if pos < idx + 1 else -1000 for pos in range(num_steps)] for idx in range(num_steps)]).unsqueeze(0) #batch x n x n mask (transpose for bmm) 
            # mask = mask.repeat(dot_product.shape[0], 1, 1).float().to(device)
            # dot_product = torch.bmm(dot_product, mask)
        out_softmax = torch.softmax(dot_product, dim=-1)
                
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
    
    def forward(self, q, k, v, mask):
        weighted_values = [] #weighted attenton
        #split and project to the hidden dim, calculate the attention and concat the attention_weighted values
        # split = q.shape[-1]//self.num_heads
        # for i in range(self.num_heads):
        #     q_i = self.hidden_q(q[:, :, i*split:(i+1)*split])
        #     k_i = self.hidden_k(k[:, :, i*split:(i+1)*split])
        #     v_i = self.hidden_v(v[:, :, i*split:(i+1)*split])
        #     weighted_values.append(attention_score(q_i, k_i, v_i, mode=mode))
        q_i = self.hidden_q(q.view(q.shape[0], self.num_heads, q.shape[1], q.shape[-1]//self.num_heads))
        k_i = self.hidden_k(k.view(k.shape[0], self.num_heads,  k.shape[1], k.shape[-1]//self.num_heads))
        v_i = self.hidden_v(v.view(v.shape[0], self.num_heads, v.shape[1], v.shape[-1]//self.num_heads))

        print("QKV", q_i.shape, k_i.shape, v_i.shape)
        print('mask', mask.shape)
        dot_product = torch.matmul(q_i, k_i.transpose(2, 3))/q_i.shape[-1]**0.5 # b x h x m x n
        dot_product = dot_product.masked_fill(mask!=0, -10000) #mask shape 1 x 1 x 1 x n
        weighted_values = torch.matmul(F.softmax(dot_product, -1), v_i) #b x h x m x f 
        weighted_values = weighted_values.reshape(weighted_values.shape[0], weighted_values.shape[2], self.num_heads*weighted_values.shape[-1])  #b x m x f 
        return self.out(weighted_values)



#layer norm skip connection 
def AddNorm(x1, x2, dropout=0.1):
    #x1 is the attention
    layernorm = nn.LayerNorm(x1.shape[-1]).to(device)
    dropout = nn.Dropout(dropout).to(device)
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
    q = torch.rand([32, 12, 128])
    k = torch.rand([32, 12, 128])
    v = torch.rand([32, 12, 128])
    # valid_lens = torch.tensor(np.random.randint(1, 9, 32)) #batch 
    # print('batch valid lens', valid_lens.shape)
    # print('Attention score', attention_score(q, k, v, mode='encoder').shape)
    mask = torch.rand([32, 1, 1, 12])
    # #multi head
    attention = Multi_head_attention(128, 256, 128)
    print('multi head', attention(q, k, v, mask).shape) #batch x nstep x feature 
    # print('layernorm', AddNorm(q, q).shape)
