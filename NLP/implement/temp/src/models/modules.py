#py
import sys
from einops import repeat
import math


#dllib
import torch
import torch.nn as nn
import torch.nn.functional as F


#local (need to add local dir to path for importing)
sys.path.append('/home/xps/projects/deep-learning-/NLP/implement/temp/src')
from utils import AttentionWeight

'''
- method: __call__()
A() -> A(input) == A.forward(input), A will have data&behaviors of Pytorch Module class 
- Multi-head : h = 8, d = 512 
'''
#Positional Encoding
def PositionalEncoding(type, scale, p_drop, plot=False, **kwargs):
    '''
    - Relative PE vs Trigonometric PE
    Trigonometric PE: sin for 2i and cos for 2i + 1 -> n x d <pos, dim> 

    - kwargs: dict, args: list input stream
    - Dropout for embedding
    '''
    if type == "direct":
        x, d_model = kwargs.values()
        PE = torch.zeros(x.shape[1], x.shape[2])
        for i in range(PE.shape[0]):
            for j in range(PE.shape[1]):
                if j % 2 == 0:
                    PE[i, j] += math.sin(i/scale**(j/d_model))
                else:
                    PE[i, j] += math.cos(i/scale**((j-1)/d_model))
        
        x += PE
        x = nn.Dropout(p_drop)(x)
        if plot:
            return x, PE
        return x

    if type == "relative":
        pass


#Point-wise FFN
class PointwiseFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.W_ff = nn.Linear(d_model, d_ff)
        self.W_o = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.W_o(F.relu(self.W_ff(x)))
#Multihead attnetion
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8, attention='dot', mask=None):
        
        '''
        - num_heads, d_model -> Projection Q, K, V subspaces
        - attention weight -> Learning attention
        - mask
        - q, k, v: model inputs
        '''
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.attention = attention
        self.mask = mask
        self.d_head = int(d_model/h)
        self.W_q = nn.Linear(d_model, self.d_head)
        self.W_k = nn.Linear(d_model, self.d_head)
        self.W_v = nn.Linear(d_model, self.d_head)
        self.W_o = nn.Linear(d_model, d_model)



    def forward(self, q, k, v):
        '''
        split b x n x h x d_model -> project +  8 heads: d_head -> concat b x n x d_model -> spaces projection
        '''
        #split 
        q = repeat(q, 'b n d -> b h n d', h=8)
        k = repeat(k, 'b n d -> b h n d', h=8)
        v = repeat(v, 'b n d -> b h n d', h=8)
        q, k, v = q.reshape(-1, q.shape[2], self.d_model), k.reshape(-1, k.shape[2], self.d_model), v.reshape(-1, v.shape[2], self.d_model)
        #project + attention 
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        output = AttentionWeight(q, k, v, type=self.attention, mask=self.mask)
        #concat heads + proj to original (Same shape notwithstandingsss)
        output = output.reshape(-1, output.shape[1], self.d_model)
        output = self.W_o(output)
        

        return output

class EncoderBlock(nn.Module):
    '''
    - Multi head attention 
    - LayerNorm
    - Dropout
    - Residual add
    - Mask: transparent (ones)
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.n_seq, self.d_model, self.d_ff, self.h, self.N, self.p_drop, _ = kwargs.values()
        self.mask = torch.ones(self.n_seq, self.n_seq)
        self.ffn = PointwiseFFN(d_model=self.d_model, d_ff=self.d_ff)
        self.attention = MultiHeadAttention(d_model=self.d_model, h=self.h, mask=self.mask)
    def forward(self, x):
        output = self.addNorm(self.attention(x, x, x))
        output = self.addNorm(self.ffn(output))
        return output


    def addNorm(self, x):
        return nn.Dropout(self.p_drop)(nn.LayerNorm(x.shape[-1])(x)) + x

class DecoderBlock(nn.Module):
    '''
    - Multi head attention 
    - LayerNorm
    - Dropout
    - Residual add
    - n_seq here is len seq generated (Traing avaiable, Testing set by user)
    - Mask: transparent(Encoder outputs key), Autoregressive(Decoder output key)
    - N_seq output: all sequences (Learning), concat with past block (Inference)
    - state contains the encoder info, and the latest POSITION output for concatenation at block i(key)
        state[0]: encoder ouput, state[1]: encoder n_seq, state[2]: <pos - 1, i> 
    '''
    def __init__(self, i, mode='training', **kwargs):
        super().__init__()
        self.i = i
        self.n_seq, self.d_model, self.d_ff, self.h, self.N, self.p_drop, _ = kwargs.values()
        self.mode = mode
        if mode == "training":
            self.mask = torch.tril(torch.ones(self.n_seq, self.n_seq))
        self.ffn1 = PointwiseFFN(d_model=self.d_model, d_ff=self.d_ff)
        self.mask_attention = MultiHeadAttention(d_model=self.d_model, h=self.h, mask=self.mask)
        self.ffn2 = PointwiseFFN(d_model=self.d_model, d_ff=self.d_ff)
        self.attention = MultiHeadAttention(d_model=self.d_model, h=self.h, mask=self.mask)
    def forward(self, x, state):
        #state[2][i] is block i for latest position
        i = self.i
        print("STATE", len(state))
        if state[2][i] == None:
            state[2][i] = x
        else:
            state[2][i] = torch.cat([state[2][i], x], dim=1)
        if self.mode == "training":
            k = x
        else:
            k = state[2][i]
            
        x = self.mask_attention(x, k, k)
        x = self.addNorm(x)
        x = self.attention(x, state[0], state[0])
        output = self.addNorm(x)
        return output, state


    def addNorm(self, x):
        return nn.Dropout(self.p_drop)(nn.LayerNorm(x.shape[-1])(x)) + x
        
class Encoder(nn.Module):
    '''
    same q, k, v
    N: num_blocks
    heads: num_heads
    n_seq: len seq
    d_model:
    p_dropout:
    label_smoothing:
    '''
    def __init__(self,  **kwargs):
        super().__init__()
        self.n_seq, self.d_model, self.d_ff, self.h, self.N, self.p_drop, self.label_smoothing = kwargs.values()
        self.Embedding = nn.Embedding(self.n_seq, self.d_model)
        moduleList = [EncoderBlock(**kwargs) for i in range(self.N)]
        self.layers = nn.Sequential(*moduleList)

    def forward(self, x):
        x = self.Embedding(x)
        x = PositionalEncoding('direct', scale=10000, p_drop=self.p_drop, x=x, d_model=self.d_model)
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    '''
    - 

    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.n_seq, self.d_model, self.d_ff, self.h, self.N, self.p_drop, self.label_smoothing = kwargs.values()
        self.Embedding = nn.Embedding(self.n_seq, self.d_model)
        moduleList = [DecoderBlock(i, **kwargs) for i in range(self.N)]
        self.layers = nn.Sequential(*moduleList)
    def init_state(self, encoder_output, encoder_n_seq):
        return [encoder_output, encoder_n_seq, [None]*self.N]
    def forward(self, x, state):
        x = self.Embedding(x)
        x = PositionalEncoding('direct', scale=10000, p_drop=self.p_drop, x=x, d_model=self.d_model)
        for layer in self.layers:
            x, state = layer(x, state)
            print(x.shape, layer.i)
        return x, state
            






class Seq2Seq(nn.Module):
    #aggregation OOP
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        '''
        output when testing: '<sos>'
        '''
        x = self.encoder(x)
        print("????", x.shape)
        #encoder: b x n x d
        state = self.decoder.init_state(x, x.shape[1])
        print("XXXX", state[0].shape, state[1])
        #decoder:
        outputs, state = self.decoder(y, state)
        print('????/')
        return outputs, state



if __name__ == '__main__':
    '''
    Testing the modules 
    '''
    q, k, v = torch.rand(32, 5, 512), torch.rand(32, 5, 512), torch.rand(32, 5, 512)
    print("bf PE", q[0, 3, 9])
    print("Attention Block", EncoderBlock(n_seq=5, d_model=512, d_ff=2048, h=8, N=6, p_drop=0.1, label_smoothing=None)(q).shape)
    q = PositionalEncoding('direct', 10000, 0.1, x=q, d_model=q.shape[-1])
    print("af PE", q[0, 3, 9])
    # Encoder
    # self.n_seq, self.d_model, self.d_ff, self.h, self.N, self.p_drop, self.label_smoothing
    x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long).unsqueeze(0)
    encoder = Encoder(n_seq=5, d_model=512, d_ff=2048, h=8, N=6, p_drop=0.1, label_smoothing=None)
    print("Encoder", encoder(x).shape)
    decoder = Decoder(n_seq=5, d_model=512, d_ff=2048, h=8, N=6, p_drop=0.1, label_smoothing=None)
    seq2seq = Seq2Seq(encoder, decoder)
    #rand 
    x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long).unsqueeze(0)
    y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long).unsqueeze(0)
    print("seq2seq", seq2seq(x, y)[0].shape)