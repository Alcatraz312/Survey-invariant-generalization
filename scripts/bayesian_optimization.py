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
from scripts.train import prepare_dataloader

import h5py
from tqdm import tqdm

import os
import gc

import optuna



def training_step(model, batch, optimizer, device, beta, loss_agg):

    ''' 
    Training of a batch exactly once \n
    Parameters : \n model : Torch model\n
    batch : single data batch \n
    optimizer : torch optimizer for loss propagation \n
    device : cuda or cpu for matrix multiplication \n
    beta : Kl divergence weight \n
    loss_agg : u or s (type of loss , u for uncertainty aggregation, s for sum aggregation)
    '''

    model.train()    # training mode

    flux, y_reg, y_cls = [b.to(device) for b in batch]   # load the batch into cuda

    optimizer.zero_grad()    # zero the gradient
    out = model(flux)     # feed forward the flux data

    # aggregating losses
    if loss_agg.lower() == "u":
        loss, components = model.uncertainty_aggregate_loss(
            x = flux,
            x_hat = out["x_hat"],
            mu = out["mu"],
            logvar = out["logvar"],
            mu_reg = out["mu_reg"],
            logvar_reg = out["logvar_reg"],
            y = y_reg,
            cls_logits = out["cls_logits"],
            labels = y_cls,
            beta = beta        
        )
    elif loss_agg.lower() == "s":
        loss, components = model.sum_aggregate_loss(
            x = flux,
            x_hat = out["x_hat"],
            mu = out["mu"],
            logvar = out["logvar"],
            mu_reg = out["mu_reg"],
            logvar_reg = out["logvar_reg"],
            y = y_reg,
            cls_logits = out["cls_logits"],
            labels = y_cls,
            beta = beta        
        )

    loss.backward()    # back propagation

    # Gradient clipping — prevents exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # prevents gradient overshoot 

    optimizer.step()    # take a descent step
    return components
    
@torch.no_grad()
def validation_step(model, batch, device, beta, loss_agg):

    ''' 
    Validation of a batch exactly once \n
    Parameters : \n model : Torch model\n
    batch : single data batch \n
    device : cuda or cpu for matrix multiplication \n
    beta : Kl divergence weight
    '''
    
    model.eval()   # validation mode

    flux, y_reg, y_cls = [b.to(device) for b in batch]
    out = model(flux) 

    if loss_agg == "u":
        loss, components = model.uncertainty_aggregate_loss(
            x = flux,
            x_hat = out["x_hat"],
            mu = out["mu"],
            logvar = out["logvar"],
            mu_reg = out["mu_reg"],
            logvar_reg = out["logvar_reg"],
            y = y_reg,
            cls_logits = out["cls_logits"],
            labels = y_cls,
            beta = beta        
        )
    elif loss_agg == "s":
        loss, components = model.sum_aggregate_loss(
            x = flux,
            x_hat = out["x_hat"],
            mu = out["mu"],
            logvar = out["logvar"],
            mu_reg = out["mu_reg"],
            logvar_reg = out["logvar_reg"],
            y = y_reg,
            cls_logits = out["cls_logits"],
            labels = y_cls,
            beta = beta        
        )

    # classification accuracy
    preds = out["cls_logits"].argmax(dim = 1)    # predicted class indices 

    classification_accuracy_boolean = (preds == y_cls)    # element wise comparison --> boolean tensor
    classification_accuracy_float = classification_accuracy_boolean.float()        # float tensor
    classification_accuracy = classification_accuracy_float.mean().item()        # mean accuracy

    # regression MAE per parameter

    mae = (out["mu_reg"] - y_reg).abs().mean(dim = 0).cpu().numpy()   # shape --> (3,)

    return components, classification_accuracy, mae


def train(model, train_loader, val_loader, batch_size,loss_agg, n_epochs = 10, lr_tasks = 3e-4, lr_recon = 1e-4, beta = 1.0, lr_patience = 10, lr_min = 1e-5, schedule_lr = True):

    ''' 
    Training loop \n
    Parameters : \n
    model : torch model architecture \n
    train_loader = train set loader \n
    val_loader = validation set loader \n
    n_epochs = number of epochs, default set to 10 \n
    lr : learning rate of the model, defaul set to 0.001 \n
    beta : kl divergence weight
    '''
    # counter = 0

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    optimizer = torch.optim.Adam([

        # Encoder - decoder and latent space parameters 
        {"params" : model.encoder.parameters(), "lr": lr_tasks},
        {"params" : model.decoder.parameters(), "lr" : lr_recon},
        {"params" : [model.log_sigma_recon], "lr" : lr_recon},
        {"params" : model.mu_layer.parameters(), "lr" : lr_tasks},
        {"params" : model.logvar_layer.parameters(), "lr" : lr_tasks},

        # downstream task heads parameters
        {"params" : model.regression_head.parameters(), "lr" : lr_tasks},
        {"params" : model.reg_mu.parameters(), "lr" : lr_tasks},
        {"params" : model.reg_logvar.parameters(), "lr" : lr_tasks},
        {"params" : model.classification_head.parameters(), "lr" : lr_tasks},

        #homoskedastic uncertainty weights 
        {"params" : [model.logvar_reg], "lr" : lr_tasks},
        {"params" : [model.logvar_cls], "lr" : lr_tasks}
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # transfer to cuda cores

    model = model.to(device)
    # dictionary to track losses and accuracy metrics

    track_losses = {
        "train_loss" : [],
        "train_recon" : [],
        "train_kl" : [],
        "train_reg" : [],
        "train_cls" : [],
        "val_acc" : [],
        "val_mae" : [],
        "val_loss" : [],
        "val_recon" : [],
        "val_kl" : [],
        "val_reg" : [],
        "val_cls" : [],
    }

    # best_val_loss = float("inf")
    # best_epoch = 0
    # patience_count = 0

    print(f"Training on {device}")

    # training and validation loop

    # scheduling learning rate to reduce instability 
    if schedule_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",patience = lr_patience, min_lr= lr_min)

    for epoch in range(1, n_epochs + 1):

        # beta = get_beta(epoch, warmup_epochs=30, beta_max=0.5)     # beta annealing
        # exp.log_metric("beta", beta, epoch=epoch) 

        # traning:

        train_components = {"loss" : 0, "recon" : 0, "kl" : 0, "reg" : 0, "cls" : 0}    # initialize loss dictionary for training for each epoch
        n_train_batches = 0           # number of batches in the training 

        for batch in train_loader:
            components = training_step(model, batch, optimizer, device, beta, loss_agg)     # training the model
            for k in train_components:
                train_components[k] += components[k]       # initializing the values into the loss dictionary

            n_train_batches += 1

        train_avg = {k : v/n_train_batches for k,v in train_components.items()}     # average the losses over all batches for one epoch 

        # validation

        val_components = {"loss" : 0, "recon" : 0, "kl" : 0, "reg" : 0, "cls" : 0}

        # validation error metrics
        n_val_batches = 0
        total_acc = 0.0
        total_mae = np.zeros(3)
 
        for batch in val_loader:
            components, accuracy, mae = validation_step(model, batch, device, beta, loss_agg)

            for k in val_components:
                val_components[k] += components[k]

            n_val_batches += 1
            total_acc += accuracy    # total accuracy of all batches for one epoch
            total_mae += mae      # total mean absolute error of all batches for one epoch

        # averaging loss and error metrics over number of batches
        val_avg = {k : v/n_val_batches for k,v in val_components.items()}
        acc_avg = total_acc/n_val_batches
        mae_avg = total_mae/n_val_batches

        if schedule_lr:
            scheduler.step(val_avg["loss"])

    return val_avg["reg"], val_avg["cls"]

def objective(trial):

    latent_dim = trial.suggest_categorical("Latent_dim", [128, 256])

    beta = trial.suggest_categorical("beta", [1.0, 0.5, 0.1, 0.07, 0.03])

    lr_patience = trial.suggest_categorical("LR_patience", [5, 10, 20])

    batch_size = trial.suggest_categorical("Batch_size", [256, 512])

    train_loader, val_loader, test_loader = prepare_dataloader(
    flux      = flux_clean,
    y_reg     = y_reg_norm,
    y_cls     = y_cls_clean,
    batch_size = batch_size
    )

    model = MTLArchitecture(
        input_dim  = flux_clean.shape[1],
        latent_dim = latent_dim,
        num_classes = 7
)

    reg_loss, cls_loss = train(model = model, train_loader = train_loader, val_loader= val_loader, batch_size= batch_size,loss_agg= "u", n_epochs= 70, lr_tasks = 0.0003, lr_recon = 0.0001, beta = beta, lr_patience= lr_patience,
                               lr_min= 0.00001, schedule_lr= True)
    
    return reg_loss, cls_loss


def main():
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials = 15)

    # get pareto front trials
    print("Best trials:")
    for trial in study.best_trials:
        print(f"  Params: {trial.params}")
        print(f"  Values: reg={trial.values[0]:.4f}, cls={trial.values[1]:.4f}")

if __name__ == "__main__":
    main()