import torch
import torch
import torch.nn as nn 
import sys
import os 
from pathlib import Path 
import pandas as pd
import torch.nn.functional as F
from einops import repeat
import matplotlib.pyplot as plt
from pathlib import Path
PATH = Path(os.path.dirname(__file__)).parent
sys.path.append(os.path.join(PATH, 'preprocessing'))
from funcs import Vocab
from tqdm import tqdm
from neural_net import EncoderRNN, Attn, LuongAttnDecoderRNN, maskNLLLoss



def evaluate(vocab, encoder, decoder, input_variable, lengths, MAX_LENGTH=11):
    decoder_input = torch.tensor([1], dtype=torch.long).reshape(1, 1)
    encoder_output, encoder_hidden = encoder(input_variable, lengths)
    decoder_hidden = repeat(encoder_hidden[-1], 'b h -> n b h', n=decoder.n_layers)
    print(encoder_output.shape, decoder_hidden.shape)
    out_token = []
    for i in range(MAX_LENGTH):
        decoder_ouput, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
        out_token.append(torch.argmax(decoder_ouput, dim=1))
        decoder_input = torch.argmax(decoder_ouput, dim=1).reshape(1, 1)
    sentence = ""
    for token in out_token:
        sentence += vocab.idx2word[token.item()] + " "
    return sentence
def load_ckp(checkpoint, encoder, decoder, encoder_optimizer, decoder_optimizer):
    result = torch.load(checkpoint)
    encoder.load_state_dict(result['encoder'])
    decoder.load_state_dict(result['decoder'])
    encoder_optimizer.load_state_dict(result['encoder_optim'])
    decoder_optimizer.load_state_dict(result['decoder_optim'])
if __name__ == "__main__":
    #vocab:
    vocab = Vocab()
    data = pd.read_csv(os.path.join('/home/xps/educate/code/NLP/chat_bot/data/cornell movie-dialogs corpus/pair_df.csv'), sep='@')
    data = data.iloc[:3000]
    print(data)
    for i in range(len(data)):
        input, target = data.iloc[i, :]
        vocab.add_sentence(input)
        vocab.add_sentence(target)
    # Configure models
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2 
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 32 
    embedding = nn.Embedding(vocab.num_words, hidden_size)
    #define the model
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, vocab.num_words, decoder_n_layers, dropout)
    #opti
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
   
    #load and eval
    

    model_path = os.path.join(PATH, "checkpoints/checkpoint0.pt")
    word = 'hello my name is'
    word_tensor = torch.tensor([vocab.word2idx[i] for i in word.split()] + [vocab.word2idx['EOS']], dtype=torch.long) 
    if len(word.split()) < 11:
          word_tensor = torch.cat([word_tensor, torch.zeros(10 - len(input.split()), dtype=torch.long)])  
          word_tensor = word_tensor.reshape(-1, 1)
    print(word_tensor.shape)
    length = torch.tensor([len(word.split())], dtype=torch.long)
    load_ckp(model_path, encoder, decoder, encoder_optimizer, decoder_optimizer)
    print(evaluate(vocab, encoder, decoder, word_tensor, length, MAX_LENGTH=11))