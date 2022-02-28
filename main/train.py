import torch
import torch.nn as nn 
import sys
import os 
from pathlib import Path 
from dataset import Translate_dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F
from einops import repeat
from neural_net import EncoderRNN, Attn, LuongAttnDecoderRNN, maskNLLLoss
import random
import matplotlib.pyplot as plt
from pathlib import Path
PATH = Path(os.path.dirname(__file__)).parent
sys.path.append(os.path.join(PATH, 'preprocessing'))
from funcs import Vocab
from tqdm import tqdm
print("GPU", torch.cuda.is_available())

def train(input_variable, lengths, target_variable, mask, encoder, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, clip, device='cpu'):

      #input and optimizer
      if torch.cuda.is_available():
        input_variable, target_variable = input_variable.to(device), target_variable.to(device)
        encoder.cuda()
        decoder.cuda()
      decoder_input = torch.tensor([[1] for i in range(batch_size)], dtype=torch.long).reshape(1, -1)
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad() 

      #feed to model
      #encoder:
      encoder_output, encoder_hidden = encoder(input_variable, lengths)
      #repeat the final hidden according to the num_layers of encoder 
      decoder_hidden = repeat(encoder_hidden[-1], 'b h -> n b h', n=decoder.n_layers)

      total_loss = 0
      n_total = 0
      #target shape 11 x 32 (seq x batch)
      teacher_forcing = random.uniform(0, 1) > 0.5
      for i in range(target_variable.shape[0]): 
          decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
          if teacher_forcing:
            decoder_input = target_variable[i].reshape(1, batch_size) #target seq x batch
          else:
            decoder_input = torch.argmax(decoder_output, dim=1).reshape(1, batch_size) # seq 1 x batch 32 
  
          loss, ntotal = maskNLLLoss(decoder_output, target_variable[i].reshape(batch_size, 1), mask)
          total_loss += loss.item()
          n_total += ntotal
      
      #backward
      loss.backward()

      #step
      encoder_optimizer.step()
      decoder_optimizer.step()

      return total_loss/n_total

def trainiters(n_epochs, input_variable, lengths, target_variable, mask, encoder, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, clip, path, device='cpu'):
    epoch_loss = []
    for epoch in tqdm(range(n_epochs)):
        total_loss = 0
        for batch in tqdm(dataloader):
            sample_input, sample_target, lengths, mask = batch
            if sample_input.shape[0] == batch_size:
                
                sample_input = sample_input.reshape(-1, batch_size)
                sample_target = sample_target.reshape(-1, batch_size)
                total_loss += train(sample_input, lengths, sample_target, mask, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, device)
            else:
                print(sample_input.shape[0])
                continue
        epoch_loss.append(total_loss)
        state = {
            'epoch': epoch, 
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'encoder_optim': encoder_optimizer.state_dict(),
            'decoder_optim': decoder_optimizer.state_dict(), 
            'loss': total_loss
        }
        torch.save(state, f"{path}/checkpoint{epoch}.pt")

    return epoch_loss
            
           
            


        
    
      


if __name__ == '__main__':
    vocab = Vocab()
    data = pd.read_csv(os.path.join(PATH, 'data/cornell movie-dialogs corpus/pair_df.csv'), sep='@')
    data = data.iloc[:3000]
    print(data)
    for i in range(len(data)):
        input, target = data.iloc[i, :]
        vocab.add_sentence(input)
        vocab.add_sentence(target)
    dataset = Translate_dataset(vocab, data)
    print('---SHAPE', dataset[0][0].shape)
    #loader 
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    sample_input, sample_target, lengths, mask = iter(dataloader).next()
    lengths, _ = lengths.sort(descending=True)
    print('---encoder input', sample_input.shape, sample_target.shape, lengths)
    #try the model 
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
    #encode
    encoder_output, encoder_hidden  = encoder(sample_input.reshape(11, 32), lengths)
    print(encoder_output.shape, encoder_hidden.shape)
    #decoder
    decoder_input = torch.tensor([[vocab.word2idx['SOS'] for i in range(32)]])
    print('--decode input', decoder_input.shape)
    encoder_hidden = repeat(encoder_hidden[-1], 'b h -> n b h', n=decoder_n_layers)
    decoder_output, decoder_hidden = decoder(decoder_input, encoder_hidden, encoder_output)
    print('----decoder ouput', decoder_output.shape, decoder_hidden.shape)
    #sample train
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_epochs = 10
    print_every = 1
    save_every = 500
    path = os.path.join(PATH, 'checkpoints')
    sample_input = sample_input.reshape(-1, 32)
    sample_target = sample_target.reshape(-1, 32)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    print(train(sample_input, lengths, sample_target, mask, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip))
    trainiters(n_epochs, sample_input, lengths, sample_target, mask, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, path)