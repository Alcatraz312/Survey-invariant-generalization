from src.models.multitask import MTLArchitecture
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn   
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, random_split

import h5py
from tqdm import tqdm

import os
import gc
import comet_ml


def prepare_dataloader_lamost(flux, y_reg, y_cls, batch_size = 256, 
                        ):

    ''' 
    Preparing the tensor sets for flux and labels
    '''

    flux_tensor  = torch.FloatTensor(flux)    # rank 1 tensors
    y_reg_tensor = torch.FloatTensor(y_reg)
    y_cls_tensor = torch.LongTensor(y_cls)     # tensor with int values

    dataset = TensorDataset(flux_tensor, y_reg_tensor, y_cls_tensor)

    lamost_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return lamost_loader

