from encoder import *
from decoder import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import random
from example_model import *
import time
from copy import deepcopy
attention =  True
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
class Net(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding1 = nn.Embedding(vocab_size, embed_dim)
        self.embedding2 = nn.Embedding(vocab_size, embed_dim)
        self.model = nn.Transformer(d_model=embed_dim, batch_first=True)
        self.out = nn.Linear(embed_dim, vocab_size)
    def forward(self, src, trg):
        src = self.embedding1(src)
        trg = self.embedding2(trg)
        out = self.model(src, trg)
        return self.out(out)
def MaskedNLL(yhat, y, mask, mode):
    #if feed sequentially 
    # print(yhat.shape, y.shape)
    # print('BEFORE SOFTMAX', yhat)
    #mask cross entropy
    mask_target = y!=0
    # loss =  (nn.CrossEntropyLoss(reduction='none')(yhat.permute(0, 2, 1), y.squeeze(-1)) * mask_target.squeeze(-1)).mean(dim=[0, 1])
    # print('LOSS', loss)
    yhat = F.softmax(yhat, dim=-1)
    # #yhat and y for 1 timestep
    # #yhat batch  x vocab, y batch x 1 -> gather according to y[-1] and the dimension 1 (softmax dim) -> log
    # #feed in parallel: yhat batch x seq_len x vocab vs y batch x seq_len x 1
    # # for i in range(yhat.shape[1])
    # if mode == 'parralel':
    CE = -torch.log(torch.gather(yhat, 2, y))
    # else:
    #     CE = -torch.log(torch.gather(yhat, 1, y))
    # # print('orginal', CE.shape)
    # # print('log', torch.log(CE))
    # #get the mask of this time step
    CE = CE.masked_select(mask).mean()
    
    loss = CE
    return loss #for valid pos counts

    



def train(encoder, decoder, batch_data, encoder_optim, decoder_optim, mode='parralel'):
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    encoder, decoder = encoder.to(device), decoder.to(device)
    input, decoder_target, enc_valid, mask, decoder_input = batch_data
    input, decoder_target, mask, decoder_input = input.to(device), decoder_target.to(device), mask.to(device), decoder_input.to(device) 
    enc_valid = enc_valid.to(device)
    #encode the input
    enc_max_len = input.shape[1]
    # print('enc max', enc_max_len)
    # print('enc valid', enc_valid)
    
    #decoder
    num_steps = decoder_target.shape[1]
    
    #get the mask
    # decoder_input = torch.cat([torch.tensor([[1] for i in range(batch_size)], dtype=torch.long).to(device), target], dim=1)
    # decoder_target = torch.cat([t arget, torch.tensor([[2] for i in range(batch_size)], dtype=torch.long).to(device)], dim=1)
    src_mask, trg_mask = create_mask(input, decoder_input)
    #init state
    encoder_outputs = encoder(input, src_mask)
    state = decoder.init_state(encoder_outputs, enc_valid, enc_max_len)
    teacher_forcing = True if random.uniform(0, 1) > 0.5 else False
    total_loss = 0 
    valid_pos = 0
    if mode == 'parralel':
        # print('input', decoder_input[0], src_mask[0].shape, trg_mask.shape, decoder_target[0])
        pred, _ = decoder(decoder_input, state, src_mask, trg_mask, attention=attention)
        # print('shape', pred.shape, decoder_target.shape)
        loss = MaskedNLL(pred, decoder_target.unsqueeze(-1), mask.unsqueeze(-1), mode=mode)
        total_loss = loss

    else:
        decoder_input = torch.tensor([[1] for i in range(batch_size)]).to(device) #SOS tag
        for i in range(num_steps):
            src_mask, trg_mask = create_mask(input, decoder_input)
            # print('decoder input', decoder_input.shape)
            pred, state = decoder(decoder_input, state, src_mask, trg_mask)
            # print('pred', pred.shape)
            pred = F.softmax(pred, dim=-1)
            if teacher_forcing:
                decoder_input = decoder_target[:, i].unsqueeze(-1) #next time step
            else:
                decoder_input = torch.argmax(pred, dim=-1)
            loss = MaskedNLL(pred, decoder_target[:, i].unsqueeze(-1).unsqueeze(-1), mask[:, i].unsqueeze(-1).unsqueeze(-1), mode=mode)
            total_loss += loss
    total_loss.backward()
    encoder_optim.step()
    decoder_optim.step()
    return total_loss

def train2(net, batch_data, net_optim, mode='parralel'):
    net_optim.zero_grad()
    net = net.to(device)
    input, target, enc_valid, mask = batch_data
    input, target, mask = input.to(device), target[:, :-1].to(device), mask.to(device)
    # enc_valid = enc_valid.to(device)
    # #encode the input
    # enc_max_len = input.shape[1]
    # # print('enc max', enc_max_len)
    # # print('enc valid', enc_valid)
    # encoder_outputs = encoder(input, enc_valid, enc_max_len)
    #decoder
    num_steps = target.shape[1]
    # state = decoder.init_state(encoder_outputs, enc_valid, enc_max_len)
    
    
    teacher_forcing = False if random.uniform(0, 1) > 0.5 else False
    total_loss = 0 
    valid_pos = 0
    if mode == 'parralel':
        # print(target.shape)
        decoder_input = torch.cat([torch.tensor([[1] for i in range(batch_size)], dtype=torch.long).to(device), target], dim=1)
        decoder_target = torch.cat([target, torch.tensor([[2] for i in range(batch_size)], dtype=torch.long).to(device)], dim=1)
        input_mask, decoder_input_mask, decoder_target_mask = create_masks(input, decoder_input, decoder_target)
        # print('INPUT MASK', input_mask)
        # print('DECODER INPUT', decoder_input_mask)
        # print('DECODER OUTPUT', decoder_target_mask)
        pred = net(input, input_mask, decoder_input, decoder_input_mask)
        # print('shape', pred.shape, decoder_target.shape)
        loss = MaskedNLL(pred, decoder_target.unsqueeze(-1), mask.unsqueeze(-1), mode='parralel')
        total_loss = loss

    else:
        decoder_input = torch.tensor([[1] for i in range(batch_size)]).to(device) #SOS tag
        target = torch.cat([target, torch.tensor([[2] for i in range(batch_size)], dtype=torch.long).to(device)], dim=-1)
        for i in range(num_steps):
            # print('decoder input', decoder_input.shape)
            pred, state = decoder(decoder_input, state)
            # print('pred', pred.shape)
            pred = F.softmax(pred, dim=-1)
            if teacher_forcing:
                decoder_input = target[:, i].unsqueeze(-1) #next time step
            else:
                decoder_input = torch.argmax(pred, dim=-1)
            loss, valid = MaskedNLL(pred.squeeze(1), target[:, i].view(-1, 1), mask[:, i])
            total_loss += loss
            valid_pos += valid
    total_loss.backward()
    net_optim.step()
    return total_loss




class Translate_dataset(Dataset):
    def __init__(self, vocab, data, MAX_LEN=10):
        super().__init__()
        self.vocab = vocab
        self.data = data
        self.MAX_LEN = MAX_LEN #max len without the EOS tag
    def __getitem__(self, idx):
        input, target = self.data.iloc[idx, :]
        #get the tensor word:
        tensor_input = torch.tensor([self.vocab.word2idx[word] for word in input.split()] + [self.vocab.word2idx['EOS']], dtype=torch.long)
        decoder_target = torch.tensor([self.vocab.word2idx[word] for word in target.split()] + [self.vocab.word2idx['EOS']], dtype=torch.long)
        decoder_input = torch.tensor([self.vocab.word2idx['SOS']] + [self.vocab.word2idx[word] for word in target.split()], dtype=torch.long)
        #get the valid_len:
        valid_len = tensor_input.shape[0] #the valid len for attention weights 
        #padding
        if len(input.split()) < self.MAX_LEN + 1:
            tensor_input = torch.cat([tensor_input, torch.zeros(self.MAX_LEN - len(input.split()), dtype=torch.long)])
        if len(target.split()) < self.MAX_LEN + 1:
            decoder_target = torch.cat([decoder_target, torch.zeros(self.MAX_LEN - len(target.split()), dtype=torch.long)])
            mask = torch.tensor([True if i.item() != 0 else False for i in decoder_target])
            decoder_input =  torch.cat([decoder_input, torch.zeros(self.MAX_LEN - len(target.split()), dtype=torch.long)])

        return tensor_input, decoder_target, valid_len, mask, decoder_input

    def __len__(self):
        return len(self.data)

class Vocab:
    def __init__(self, name=None):
        self.name = name
        self.word2idx = {"PAD": 0, "SOS": 1, "EOS": 2}
        self.idx2word = {1: "SOS", 0: "PAD", 2: "EOS"} #3 special keywords
        self.num_words = 3
        self.word_counts = {} #SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    def add_word(self, word):
        if word in self.word2idx.keys():
            self.word_counts[word] += 1

        else:
            self.word2idx[word] = self.num_words
            self.idx2word[self.num_words] = word
            self.word_counts[word] = 1 
            self.num_words += 1




if __name__ == "__main__":
    vocab = Vocab()
    data = pd.read_csv('./pair_df.csv', sep='@')
    print(data)
    for i in range(len(data)):
        input, target = data.iloc[i, :]
        for line in [input, target]:
            vocab.add_sentence(line)
    print('VOCAB', vocab.name)
    print('NUMWORDS', vocab.num_words)
    #dataset
    dataset = Translate_dataset(vocab, data)
    print('input and valid len', dataset[0][0], dataset[0][2])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sample = iter(dataloader).next()
    print('input | target | valid len | masking for output')
    print(f'{sample[0].shape} | {sample[1].shape} | {sample[2].shape} |  {sample[3].shape}')
    #NLL loss
    yhat = torch.rand([32, 11, 1000000])
    y = sample[1].unsqueeze(-1)
    mask = sample[3].unsqueeze(-1)
    
    # print('mask', MaskedNLL(yhat, y, mask, mode='none'))
    #define models
    learning_rate = 0.0001
    decoder_learning_ratio = 5
    encoder = Encoder(vocab.num_words, embed_dim=128, in_dim=128, hidden_dim=128, out_dim=128).train()
    decoder = Decoder(vocab.num_words, 128, in_dim=128, hidden_dim=128, out_dim=128).train()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    start = time.time()
    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    print('my model', train(encoder, decoder, sample, encoder_optimizer, decoder_optimizer, mode='parralel'), time.time() - start)
    #trial model in fingertips
    # net = Net(vocab_size=vocab.num_words)
    # net_optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    model = Transformer(128, 8, 4, vocab)
    model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start = time.time()
    # print('model', train2(model, sample, model_optim), time.time() - start)

    # encoder.eval()
    # decoder.eval()
    
    # #guess
    # input = 'happy birthday'
    # max_len = 11 
    # input_tensor = torch.tensor([vocab.word2idx[i] for i in input.split()] + [2], dtype=torch.long)
    # input_tensor = torch.cat([input_tensor, torch.zeros(max_len - input_tensor.shape[0], dtype=torch.long)]).view(1, -1)
    # decoder_input = torch.tensor([[1]], dtype=torch.long)
    # src_mask = input_tensor!=0
    # encoder_outputs = encoder(input_tensor, src_mask)
    # #decoder input

    # state = decoder.init_state(encoder_outputs, None, max_len)

    # answer = []
    # for i in range(max_len):
    #     print("DECODER", decoder_input)
    #     print('ENCODER', encoder_outputs.shape)
    #     size = decoder_input.shape[1]
    #     # trg_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).unsqueeze(0).unsqueeze(0)
    #     # trg_mask  = torch.ones(1, 1).unsqueeze(0).unsqueeze(0)
    #     # _, trg_mask = create_mask(input_tensor, decoder_input)

    #     pred, state = decoder(decoder_input, state, src_mask, None)
    #     # print('PRED', pred.shape)
        
    #     pred = torch.argmax(nn.functional.softmax(pred, dim=-1))
    #     # print("PRED", pred)
    #     # print('WORDS', vocab.num_words)
    #     answer.append(pred.detach().cpu().item())

    #     decoder_input =  pred.reshape(1, 1)


    # print(answer)


