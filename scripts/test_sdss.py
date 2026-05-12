from src.models.multitask import MTLArchitecture
import argparse
from src.data.data import flux_clean, y_reg_norm, y_cls_clean

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

def prepare_dataloader(flux, y_reg, y_cls, 
                       val_test_split=0.4, batch_size=256, seed=42, val_fraction = 0.25):

    ''' 
    Preparing the tensor sets for flux and labels
    '''

    flux_tensor  = torch.FloatTensor(flux)    # rank 1 tensors
    y_reg_tensor = torch.FloatTensor(y_reg)
    y_cls_tensor = torch.LongTensor(y_cls)     # tensor with int values

    # Shuffle before splitting
    n    = len(flux_tensor)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    flux_tensor  = flux_tensor[perm]
    y_reg_tensor = y_reg_tensor[perm]
    y_cls_tensor = y_cls_tensor[perm]

    dataset = TensorDataset(flux_tensor, y_reg_tensor, y_cls_tensor)

    # Compute integer sizes
    n_val  = int(n * val_fraction)
    n_test = int(n * val_test_split) - n_val  # remainder goes to test
    n_train = n - n_val - n_test

    # splitting the datasets into train, val and test datasets

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, drop_last=False)

    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear_reg", type = bool, default=False)
    parser.add_argument("--checkpoint", type=str,
                        default="/home/arbiter/projects/Survey-invariant-generalization/src/models/final_model.pth")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = prepare_dataloader(
        flux       = flux_clean,
        y_reg      = y_reg_norm,
        y_cls      = y_cls_clean,
        batch_size = 256
    )

    model = MTLArchitecture(
        input_dim   = flux_clean.shape[1],
        latent_dim  = 128,
        num_classes = 7,
        linear_reg  = args.linear_reg    # ← matches whatever was trained
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["Model_state_dict"])
    model.eval()

    all_preds_reg = []
    all_true_reg  = []
    all_preds_cls = []
    all_true_cls  = []
    all_mu = []
    all_x_hat = []
    all_flux = []

    with torch.no_grad():
        for batch in test_loader:
            flux, y_reg, y_cls = [b.to(device) for b in batch]
            
            out = model(flux)
            all_preds_reg.append(out["mu_reg"].cpu().numpy())
            all_true_reg.append(y_reg.cpu().numpy())
            all_mu.append(out["mu"].cpu().numpy())    # latent vectors

            all_x_hat.append(out["x_hat"].cpu().numpy())    # reconstructed spectrum
            all_flux.append(flux.cpu().numpy())

            cls_preds = out["cls_logits"].argmax(dim=1)
            all_preds_cls.append(cls_preds.cpu().numpy())
            all_true_cls.append(y_cls.cpu().numpy())

    # concatenate all batches
    preds_reg = np.concatenate(all_preds_reg, axis=0)  # shape (N, 3)
    true_reg  = np.concatenate(all_true_reg,  axis=0)
    preds_cls = np.concatenate(all_preds_cls, axis=0)
    true_cls  = np.concatenate(all_true_cls,  axis=0)
    all_mu = np.concatenate(all_mu, axis = 0)
    all_x_hat = np.concatenate(all_x_hat, axis = 0)
    all_flux = np.concatenate(all_flux, axis = 0)

    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/preds_reg.npy", preds_reg)
    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/true_reg.npy", true_reg)
    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/preds_cls.npy", preds_cls)
    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/true_cls.npy", true_cls)
    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/sdss_latent_mu.npy", all_mu)
    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/sdss_reconstructed_spectra.npy", all_x_hat)
    np.save("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/sdss_flux.npy", all_flux)


if __name__ == '__main__':
    main()


