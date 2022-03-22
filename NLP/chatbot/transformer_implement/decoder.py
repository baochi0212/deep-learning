import torch
import torch.nn as nn 
import numpy as np 
from neural_net import * 
VOCAB_SIZE = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"
class Decoder_block(nn.Module):
    def __init__(self, i, in_dim, hidden_dim, out_dim, num_heads=8):
        super().__init__()
        #block index -> use for previous blocks derivation (connect to the past timestep)
        self.i = i
        #attention
        self.attention1 = Multi_head_attention(in_dim, hidden_dim, out_dim, num_heads)
        self.ffn = FeedForward(in_dim, in_dim*2, in_dim)
        self.attention2 = Multi_head_attention(in_dim, hidden_dim, out_dim, num_heads)
        self.AddNorm = AddNorm(in_dim)
    def forward(self, x, state, src_mask, trg_mask=None, attention=True):
        
        #we use the last state to concat the current input
        #encoder output and valid lens
        encoder_output, enc_valid, enc_max_len = state[0], state[1], state[2]
        #get the result of last timestep 
        if state[3][self.i] is None:
            state[3][self.i] = x

        else:
            state[3][self.i] = torch.cat([state[3][self.i], x], dim=1)
        #key for autoregressive fashion
        #add norm is similar to history attention method (combine the attentioned encoder with the output of decoder)
        k = state[3][self.i]
        max_len = k.shape[1]
        valid_lens = torch.tensor([max_len for i in range(k.shape[0])])
        if self.training:
            x = self.AddNorm(self.attention1(x, k, k, trg_mask, attention), x)
        else:
            # print('KEY', k.shape)
            # print('QUERY', x.shape)

            # size = x.shape[1]
            # mask = torch.ones(size, size).unsqueeze(0).unsqueeze(0).to(device)
            x = self.AddNorm(self.attention1(x, k, k, None), x)
        x = self.AddNorm(self.attention2(x, encoder_output, encoder_output, src_mask, attention), x)
        x = self.AddNorm(self.ffn(x), x)
        return x, state

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_blocks=4, **kwargs):
        super().__init__()
        self.d_model = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pe = Positional_Encoding(embed_dim)
        
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([Decoder_block(i + 1, **kwargs) for i in range(num_blocks)])
        self.out = nn.Linear(embed_dim, vocab_size)
        
    def init_state(self, encoder_outputs, enc_valid, enc_max):
        return [encoder_outputs, enc_valid, enc_max, dict((i + 1, None) for i in range(self.num_blocks))]

    def forward(self, x, state, src_mask, trg_mask, attention=True):
        if self.training:
            attention = False
        x = self.embedding(x)*(self.d_model**0.5)
        x = x + self.pe(x)
        for i in range(self.num_blocks):
            x, state = self.blocks[i](x, state, src_mask, trg_mask, attention=attention)

            
        return self.out(x), state
        
            



if __name__ == "__main__":
    #show weight
    attention = True
    #decode each timestep in sequence: 
    block = Decoder_block(1, 128, 256, 128)
    input = torch.tensor(np.random.randint(0, 4, (1, 11)), dtype=torch.long) #batch x n_step
    #init state: 
    state = []
    state.append(torch.rand(32, 12, 128))
    state.append(torch.tensor(np.random.randint(1, 12, (32))))
    state.append(12)
    state.append({1: None})
    # print('decoder block', block(input, state, mode='decoder')[0].shape)
    # #decoder
    x = torch.tensor(np.random.randint(0, 4, (1,4)), dtype=torch.long) #batch x n_step
    decoder = Decoder(VOCAB_SIZE, 128, in_dim=128, hidden_dim=256, out_dim=128)
    state = decoder.init_state(torch.rand(1, 11, 128), torch.tensor(np.random.randint(1, 11, (32))), 12)
    src_mask, trg_mask = create_mask(input, x)
    print('decoder output', decoder(x, state, src_mask, trg_mask, attention)[0].shape) #decoder input shape =1 -> for not teacher forcing 
    #seq len: decoder_input: 1, encoder_input: 11