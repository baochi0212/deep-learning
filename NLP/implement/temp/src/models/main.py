import sys
import random
from copy import deepcopy
import pandas as pd
import os
PATH = os.environ['dir']
sys.path.append(PATH + "/src")

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
        pred = self.model(src, tgt[:, :-1])[0]

        #teacher forcing a.k.a use default ground truth (use current word -> next word) 
        if self.hparams.mode == "teacher_forcing":
            loss = MaskedNLL(pred.reshape(-1, pred.shape[-1]), tgt[:, 1:].reshape(-1))
            pred_tokens = torch.argmax(pred, dim=-1)
            self.log("train loss", loss)
            return loss
        #uniform random sampling, iterate through the n_seq of target
        else:
            decoder_input = deepcopy(tgt[:, :-1]) #sampling input 
            pred_tokens = torch.argmax(pred, dim=-1)
            for i in range(tgt[:, :-1].shape[1]):
                p = random.uniform(0, 1)
                if p > 0.69:
                    decoder_input[:, i] = pred_tokens[:, i]
            pred = self.model(src, decoder_input)[0]
            loss = MaskedNLL(pred.reshape(-1, pred.shape[-1]), tgt[:, 1:].reshape(-1))
            pred_tokens = torch.argmax(pred, dim=-1)
            self.log("train loss", loss)
            return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return {'optimizer': optimizer}
        

                    
                
if __name__ == "__main__":
    #const
    MAX_LEN = 20
    VOCAB_SIZE = len(tokenizer.vocab.keys())




    #data
    
    data = pd.read_csv(os.path.join(PATH, 'data/raw/cornell movie-dialogs corpus/pair_df.csv'), sep='@')
    print("Df", data)
    dataset = ChatDataset(data, max_length=MAX_LEN)
    src_token, tgt_token = dataset[0]
    print("Decode", tokenizer.decode(src_token))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    src, tgt = iter(dataloader).next()

    # #define lightning module 
    encoder = Encoder(vocab=VOCAB_SIZE, n_seq=MAX_LEN, d_model=512, d_ff=2048, h=8, N=6, p_drop=0.1, label_smoothing=None)
    decoder = Decoder(vocab=VOCAB_SIZE, n_seq=MAX_LEN-1, d_model=512, d_ff=2048, h=8, N=6, p_drop=0.1, label_smoothing=None)
    seq2seq = Seq2Seq(encoder, decoder)
    print("sanity check", seq2seq(src, tgt[:, :-1])[0].shape)
    plModule = ChatMachine(lr=0.01, mode='teacher_forcing')
    print("params", plModule.hparams.mode)

    #set up
    wandb_logger = WandbLogger(project='lightning_tutorial', save_dir=os.path.join(PATH, 'src/runtime'))
    ckp_dir = os.path.join(PATH, "src/runtime/checkpoints")
    custom_callbacks = [LearningRateMonitor(logging_interval='step'), EarlyStopping(monitor='val loss: ', mode='min', patience=3), ModelCheckpoint(dirpath='/media/data/chitb/study_zone/ML-_midterm_20212/final_ckp', monitor='val loss: ', mode='min')]
    #training 
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, default_root_dir=ckp_dir, logger=wandb_logger, callbacks=custom_callbacks, max_epochs=3, fast_dev_run=True)
    trainer.fit(plModule, dataloader)

        




    
