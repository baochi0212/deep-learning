import sys

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import hydra


class TranslationMachine(pl.Module):
    '''
    lightning module
    '''
