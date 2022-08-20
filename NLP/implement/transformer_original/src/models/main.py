import sys
import random
from copy import deepcopy
import pandas as pd
import os
from sklearn.model_selection import train_test_split
PATH = os.environ['dir']
sys.path.append(PATH + "/src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader



from dataset.dataset import ChatDataset
from modules import Seq2Seq, Encoder, Decoder
from utils import MaskedNLL
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
special_tokens_dict = {'additional_special_tokens': ['[SOS]','[EOS]']}
tokenizer.add_special_tokens(special_tokens_dict)

'''
- pl_module
- simple torch script for debug
''' 

class ChatMachine(pl.LightningModule):
    '''
    lightning module for dataloader, training and testing
    '''
    def __init__(self, lr, mode='teacher_forcing'):
        super().__init__()
        self.encoder = Encoder(vocab=VOCAB_SIZE, n_seq=MAX_LEN, d_model=512, d_ff=2048, h=8, N=6, p_drop=0.1, label_smoothing=None)
        self.decoder = Decoder(vocab=VOCAB_SIZE, n_seq=MAX_LEN-1, d_model=512, d_ff=2048, h=8, N=6, p_drop=0.1, label_smoothing=None)
        self.model = Seq2Seq(self.encoder, self.decoder)
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        src, tgt = batch
        decoder_input, decoder_output = deepcopy(tgt[:, :-1]), deepcopy(tgt[:, 1:])
        pred = self.model(src, decoder_input)[0]

        #teacher forcing a.k.a use default ground truth (use current word -> next word) 
        if self.hparams.mode == "teacher_forcing":
            loss = MaskedNLL(pred.reshape(-1, pred.shape[-1]), decoder_output.reshape(-1))
            pred_tokens = torch.argmax(pred, dim=-1)
            self.log("train loss", loss)
            return loss
        #uniform random sampling, iterate through the n_seq of target
        else:
            pred_tokens = torch.argmax(pred, dim=-1)
            for i in range(tgt[:, :-1].shape[1]):
                p = random.uniform(0, 1)
                if p > 0.69:
                    decoder_input[:, i] = pred_tokens[:, i]
            pred = self.model(src, decoder_input)[0]
            loss = MaskedNLL(pred.reshape(-1, pred.shape[-1]), decoder_output.reshape(-1))
            pred_tokens = torch.argmax(pred, dim=-1)
            self.log("train loss", loss)
            return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return {'optimizer': optimizer}


def train(train_dataloader, val_dataloader, model, mode='teacher_forcing', device = 'cuda' if torch.cuda.is_available() else 'cpu', lr=5e-6):
    total_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    for i in range(3):
        for src, tgt in train_dataloader:
            optimizer.zero_grad()
            src, tgt = src.to(device), tgt.to(device)
            decoder_input, decoder_output = deepcopy(tgt[:, :-1]), deepcopy(tgt[:, 1:])
            pred = model(src, decoder_input)[0]
            if mode == "teacher_forcing":
                loss = MaskedNLL(pred.reshape(-1, pred.shape[-1]), decoder_output.reshape(-1))
                pred_tokens = torch.argmax(pred, dim=-1)
            #uniform random sampling, iterate through the n_seq of target
            else:
                pred_tokens = torch.argmax(pred, dim=-1)
                for i in range(tgt[:, :-1].shape[1]):
                    p = random.uniform(0, 1)
                    if p > 0.69:
                        decoder_input[:, i] = pred_tokens[:, i]
                pred = model(src, decoder_input)[0]
                loss = MaskedNLL(pred.reshape(-1, pred.shape[-1]), decoder_output.reshape(-1))
                pred_tokens = torch.argmax(pred, dim=-1)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            if len(total_loss) % 50 == 0:
                print("Loss: ", loss.item())
    return total_loss



        

                    
                
if __name__ == "__main__":
    #const
    MAX_LEN = 20
    VOCAB_SIZE = len(tokenizer.vocab.keys())




    #data
    
    data = pd.read_csv(os.path.join(PATH, 'data/raw/cornell movie-dialogs corpus/pair_df.csv'), sep='@')
    print("Df", data)
    train_data, val_data = train_test_split(data, test_size=0.3)
    train_dataset = ChatDataset(train_data, max_length=MAX_LEN)
    val_dataset = ChatDataset(val_data, max_length=MAX_LEN)
    src_token, tgt_token = train_dataset[0]
    print("Decode", tokenizer.decode(src_token))
    # train_dataset, val_dataset = train_test_split(dataset, test_size=0.3)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)
    src, tgt = iter(train_dataloader).next()

    # #define lightning module 
    encoder = Encoder(vocab=VOCAB_SIZE, n_seq=MAX_LEN, d_model=256, d_ff=64, h=8, N=6, p_drop=0.1, label_smoothing=None)
    decoder = Decoder(vocab=VOCAB_SIZE, n_seq=MAX_LEN-1, d_model=256, d_ff=64, h=8, N=6, p_drop=0.1, label_smoothing=None)
    seq2seq = Seq2Seq(encoder, decoder)
    print("sanity check", seq2seq(src, tgt[:, :-1])[0].shape)
    plModule = ChatMachine(lr=0.01, mode='teacher_forcing')
    print("params", plModule.hparams.mode)
    pred, state = seq2seq(src, tgt[:, :-1])
    print("Loss", MaskedNLL(pred.reshape(-1, pred.shape[-1]), tgt[:, 1:].reshape(-1)))
    print(len(state[2]))
    # #set up
    # wandb_logger = WandbLogger(project='lightning_tutorial', save_dir=os.path.join(PATH, 'src/runtime'))
    # ckp_dir = os.path.join(PATH, "src/runtime/checkpoints")
    # custom_callbacks = [LearningRateMonitor(logging_interval='step'), EarlyStopping(monitor='val loss: ', mode='min', patience=3), ModelCheckpoint(dirpath='/media/data/chitb/study_zone/ML-_midterm_20212/final_ckp', monitor='val loss: ', mode='min')]
    # #training 
    # trainer = pl.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu',  default_root_dir=ckp_dir, logger=wandb_logger, callbacks=custom_callbacks, max_epochs=3, fast_dev_run=True)
    # trainer.fit(plModule, dataloader)
    # train(train_dataloader, val_dataloader, seq2seq)
        




    
