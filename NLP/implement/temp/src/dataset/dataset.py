import os
import pandas as pd
import numpy as np



import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
special_tokens_dict = {'additional_special_tokens': ['[SOS]','[EOS]']}
tokenizer.add_special_tokens(special_tokens_dict)
VOCAB_SIZE = tokenizer.vocab.keys()
'''
use pretrained bert tokenizer for english dataset
special tokens: [PAD]:0 
'''

class ChatDataset(Dataset):
    def __init__(self, data, batch=32, max_length=20):
        super().__init__()
        self.data = data
        self.max_length = max_length
        self.batch = batch
    def __getitem__(self, idx):
        src, tgt = self.data.iloc[idx, :]
        src = '[SOS] ' + src + ' [EOS]'
        tgt = '[SOS] ' + tgt + ' [EOS]'
        
        #get the tensor word:
        #drop the [CLS] token and [SEP]
        #current tokenizer, MAX_LEN + 2 -> drop CLS and SEP token later ? ``
        while len(tokenizer(src)['input_ids']) < self.max_length + 2:
            src += ' [PAD]'
        while len(tokenizer(tgt)['input_ids']) < self.max_length + 2:
            tgt += ' [PAD]'
        src_token = torch.tensor(tokenizer(src)['input_ids'], dtype=torch.long)
        tgt_token = torch.tensor(tokenizer(tgt)['input_ids'], dtype=torch.long)

        if len(src_token) > self.max_length + 2:
            src_token = src_token[:self.max_length + 2]
        if len(tgt_token) > self.max_length + 2:
            tgt_token = tgt_token[:self.max_length + 2]
        src_token, tgt_token = src_token[1:-1], tgt_token[1:-1]   
       




        return src_token, tgt_token

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    PATH = os.environ['dir']
    data = pd.read_csv(os.path.join(PATH, 'data/raw/cornell movie-dialogs corpus/pair_df.csv'), sep='@')
    print("Df", data)
    dataset = ChatDataset(data)
    src_token, tgt_token = dataset[0]
    print("Decode", tokenizer.decode(src_token))
    for i in range(len(dataset)):
        src, tgt = dataset[i]
        if src.shape[-1] != 20 or tgt.shape[-1] != 20:
            print(i, src.shape, tgt.shape)
            break