from torch.utils.data import Dataset, DataLoader
import torch
import sys
from pathlib import Path
import os 
import torch
import pandas as pd

PATH = Path(os.path.dirname(__file__)).parent
sys.path.append(os.path.join(PATH, 'preprocessing'))
from funcs import Vocab
#add the subfolder not the modules
#we use <EOS> for input and target_decode, while <SOS> for the target_encode
class Translate_dataset(Dataset):
    def __init__(self, vocab, data, batch=32, MAX_LEN=10):
        super().__init__()
        self.vocab = vocab
        self.data = data
        self.MAX_LEN = MAX_LEN 
        self.batch = batch
    def __getitem__(self, idx):
        input, target = self.data.iloc[idx, :]
        #get the tensor word:
        tensor_input = torch.tensor([self.vocab.word2idx[word] for word in input.split()] + [self.vocab.word2idx['EOS']], dtype=torch.long)
        tensor_target = torch.tensor([self.vocab.word2idx[word] for word in target.split()] + [self.vocab.word2idx['EOS']], dtype=torch.long)
        item = [tensor_input, tensor_target] #just a copy not affect 2 above variables 
        #padding
        if len(input.split()) < self.MAX_LEN + 1:
            tensor_input = torch.cat([tensor_input, torch.zeros(self.MAX_LEN - len(input.split()), dtype=torch.long)])
        if len(target.split()) < self.MAX_LEN + 1:
            tensor_target = torch.cat([tensor_target, torch.zeros(self.MAX_LEN - len(target.split()), dtype=torch.long)])
            mask = torch.tensor([True if i.item() != 0 else False for i in tensor_target])

        return tensor_input, tensor_target, tensor_input.shape[0], mask

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    print('DATASET')
    vocab = Vocab()
    data = pd.read_csv(os.path.join(PATH, 'data/cornell movie-dialogs corpus/pair_df.csv'), sep='@')
    print(data)
    for i in range(len(data)):
        input, target = data.iloc[i, :]
        vocab.add_sentence(input)
        vocab.add_sentence(target)
    dataset = Translate_dataset(vocab, data)
    dataloader = DataLoader(dataset, batch_size=32)
    sample_input, sample_target, lengths, mask = iter(dataloader).next()
    print(sample_input.shape, sample_target.shape, lengths, mask.shape)
    print('MASK', dataset[19849][-1])
    
    