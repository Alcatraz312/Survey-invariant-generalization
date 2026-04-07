from src.models.multitask import MTLArchitecture
import argparse
from src.data.lamost_data import flux_clean, y_reg_norm, y_cls_clean

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
                        seed=42):

    ''' 
    Preparing the tensor sets for flux and labels
    '''

    flux_tensor  = torch.FloatTensor(flux)    # rank 1 tensors
    y_reg_tensor = torch.FloatTensor(y_reg)
    y_cls_tensor = torch.LongTensor(y_cls)     # tensor with int values

    dataset = TensorDataset(flux_tensor, y_reg_tensor, y_cls_tensor)

    lamost_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return lamost_loader


def main():
    lamost_loader = prepare_dataloader_lamost(
        flux = flux_clean, 
        y_reg = y_reg_norm,
        y_cls = y_cls_clean,
        batch_size= 256
    )

    model = MTLArchitecture(
            input_dim   = flux_clean.shape[1],
            latent_dim  = 256,
            num_classes = 7
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # transfer to cuda cores

    model = model.to(device)

    checkpoint = torch.load("/home/arbiter/projects/Survey-invariant-generalization/src/models/final_model.pth")
    model.load_state_dict(checkpoint["Model_state_dict"])
    model.eval()

    all_preds_reg = []
    all_true_reg  = []
    all_preds_cls = []
    all_true_cls  = []

    with torch.no_grad():
        for batch in lamost_loader:
            flux, y_reg, y_cls = [b.to(device) for b in batch]
            
            out = model(flux)
            all_preds_reg.append(out["mu_reg"].cpu().numpy())
            all_true_reg.append(y_reg.cpu().numpy())

            cls_preds = out["cls_logits"].argmax(dim=1)
            all_preds_cls.append(cls_preds.cpu().numpy())
            all_true_cls.append(y_cls.cpu().numpy())

    # concatenate all batches
    preds_reg = np.concatenate(all_preds_reg, axis=0)  # shape (N, 3)
    true_reg  = np.concatenate(all_true_reg,  axis=0)
    preds_cls = np.concatenate(all_preds_cls, axis=0)
    true_cls  = np.concatenate(all_true_cls,  axis=0)

    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/lamost_preds_reg.npy", preds_reg)
    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/lamost_true_reg.npy", true_reg)
    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/lamost_preds_cls.npy", preds_cls)
    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/lamost_true_cls.npy", true_cls)


if __name__ == "__main__":
    main()