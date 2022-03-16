from neural_net import *
import numpy as np


VOCAB_SIZE = 10000
class Encoder_block(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8):
        super().__init__()

        #main
        self.attention1 = Multi_head_attention(in_dim, hidden_dim, out_dim, num_heads)
        self.ffn = FeedForward(in_dim, 2*in_dim, in_dim)
        self.attention2 = Multi_head_attention(in_dim, hidden_dim, out_dim, num_heads)

    def forward(self, x, valid_lens, max_len):
        
        #main
        x = AddNorm(self.attention1(x, x, x, valid_lens, max_len), x)
        x = self.ffn(x)
        x = AddNorm(self.attention2(x, x, x, valid_lens, max_len), x)
        return x 

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_blocks=4, **kwargs):
        super().__init__()
        #positional encoding + embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pe = Positional_Encoding(embed_dim)
        
        self.num_blocks = num_blocks
        self.block = Encoder_block(**kwargs)

    def forward(self, x, valid_lens, max_len):
        #embedding
        x = self.embedding(x)
        x = x + self.pe(x)
        for i in range(self.num_blocks):
            x = self.block(x, valid_lens, max_len)
        return x 

    @property
    def attention_weights(self):
        pass


if __name__ == "__main__":
    #encoder: we put whole tensor to encode 
    input = torch.tensor(np.random.randint(0, 100, (32, 11)), dtype=torch.long) #batch x n_step
    valid_lens = torch.tensor(np.random.randint(1, 12, (32))) #batch
    print('valid lens', valid_lens)
    max_len = input.shape[1]
    block = Encoder_block(128, 256, 128)
    block_input = torch.rand(32, 12, 128)
    print('encoder block', block(block_input, valid_lens, max_len).shape)
    encoder = Encoder(VOCAB_SIZE, embed_dim=128, in_dim=128, hidden_dim=256, out_dim=128)
    print('encoder output', encoder(input, valid_lens, max_len).shape)