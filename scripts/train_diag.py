from src.models.multiprocess import VAE, DownHeads
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

comet_key = os.environ.get("comet_ml_key")


def prepare_dataloader(flux, y_reg, y_cls, 
                       val_test_split=0.4, batch_size=256, seed=42):

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
    n_val_test = int(n * val_test_split)
    n_train    = n - n_val_test
    n_val      = n_val_test // 2
    n_test     = n_val_test - n_val    # handles odd numbers cleanly

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

def training_step_vae(model, batch, optimizer, device, beta):

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
    recon = model.reconstruction_loss(x=flux, x_hat=out["x_hat"])
    kl    = model.kl_divergence(mu=out["mu"], logvar=out["logvar"])
    loss  = recon + beta * kl

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {"vae_loss": loss.item(), "recon": recon.item(), "kl": kl.item()}
    
@torch.no_grad()
def validation_step_vae(model, batch, device, beta):

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

    recon = model.reconstruction_loss(x=flux, x_hat=out["x_hat"])
    kl    = model.kl_divergence(mu=out["mu"], logvar=out["logvar"])
    loss  = recon + beta * kl

    return {"vae_loss": loss.item(), "recon": recon.item(), "kl": kl.item()}

def training_step_tasks(model, batch, optimizer, device):

    model.train()
    flux, y_reg, y_cls = [b.to(device) for b in batch]

    optimizer.zero_grad()
    out = model(flux)

    reg  = model.regression_loss(out["mu_reg"], out["logvar_reg"], y_reg)
    cls  = model.classification_loss(out["cls_logits"], y_cls)
    loss = reg + cls

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {"loss": loss.item(), "reg": reg.item(), "cls": cls.item()}

@torch.no_grad()
def validation_step_tasks(model, batch, device):

    model.eval()
    flux, y_reg, y_cls = [b.to(device) for b in batch]
    out = model(flux)

    reg  = model.regression_loss(out["mu_reg"], out["logvar_reg"], y_reg)
    cls  = model.classification_loss(out["cls_logits"], y_cls)
    loss = reg + cls

    preds    = out["cls_logits"].argmax(dim=1)
    accuracy = (preds == y_cls).float().mean().item()
    mae      = (out["mu_reg"] - y_reg).abs().mean(dim=0).cpu().numpy()

    return {"loss": loss.item(), "reg": reg.item(), "cls": cls.item()}, accuracy, mae

def train(model, train_loader, val_loader, batch_size, n_epochs = 10, lr = 3e-4, beta = 1.0):  

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # transfer to cuda cores

    model = model.to(device)

    exp = comet_ml.start(api_key = comet_key, project_name = "Dissertation ML")
    exp.set_name(f" Multiprocess Experiment Batch Size :{batch_size} , lr : {lr} , epochs : {n_epochs}")

    exp.log_parameters({
        "n_epochs"   : n_epochs,
        "lr" : lr,
        "beta"       : beta,
        "latent_dim" : model.mu_layer.out_features,
        "input_dim"  : model.encoder[0].in_features,
        "batch_size" : train_loader.batch_size,
    })

    track_losses = {
        "train_vae_loss" : [],
        "train_recon" : [],
        "train_kl" : [],
        "train_reg" : [],
        "train_cls" : [],
        "val_acc" : [],
        "val_mae" : [],
        "val_vae_loss" : [],
        "val_recon" : [],
        "val_kl" : [],
        "val_reg" : [],
        "val_cls" : [],
    }

    print(f"Training on {device}")

    for epoch in range(1, n_epochs + 1):

        vae_train_components = {"vae_loss" : 0, "recon" : 0, "kl" : 0}
        vae_train_batches = 0

        for batch in train_loader:
            vae_components = training_step_vae(model, batch, optimizer, device, beta)
            for k in vae_train_components:
                vae_train_components[k] += vae_components[k]

            vae_train_batches += 1

        vae_train_avg = {k: v/vae_train_batches for k,v in vae_train_components.items()}


        vae_val_components = {"vae_loss" : 0, "recon" : 0, "kl" : 0}

        vae_val_batches = 0

        for batch in val_loader:
            vae_components = validation_step_vae(model, batch, device, beta)
            for k in vae_val_components:
                vae_val_components[k] += vae_components[k]

            vae_val_batches += 1

        vae_val_avg = {k: v/vae_val_batches for k,v in vae_val_components.items()}


        track_losses["train_vae_loss"].append(vae_train_avg["vae_loss"])
        track_losses["train_recon"].append(vae_train_avg["recon"])
        track_losses["train_kl"].append(vae_train_avg["kl"])
        
        track_losses["val_vae_loss"].append(vae_val_avg["vae_loss"])
        track_losses["val_recon"].append(vae_val_avg["recon"])
        track_losses["val_kl"].append(vae_val_avg["kl"])

        exp.log_metrics(
            {
                # Train VAE losses
                "Train/VAE_loss" : vae_train_avg["vae_loss"],
                "Train/Reconstruction" : vae_train_avg["recon"],
                "Train/KL" : vae_train_avg["kl"],

                # Validation VAE losses
                "Validation/VAE_loss" : vae_val_avg["vae_loss"],
                "Validation/Reconstruction" : vae_val_avg["recon"],
                "Validation/KL" : vae_val_avg["kl"]
            }, epoch= epoch
        )

    # freezing the encoder for tasks
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.mu_layer.parameters():
        param.requires_grad = False
    for param in model.logvar_layer.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters(): 
        param.requires_grad = False

    # New optimizer — only trains unfrozen parameters
    task_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    print("Encoder frozen — training task heads only")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(task_optimizer, "min", factor= 0.5, patience = 20)

    for epoch in range(1, n_epochs + 1):

        # ── Task train ─────────────────────────────────────────────
        tasks_train_components = {"loss": 0, "reg": 0, "cls": 0}
        tasks_train_batches    = 0

        for batch in train_loader:
            tasks_components = training_step_tasks(model, batch, task_optimizer, device)
            for k in tasks_train_components:
                tasks_train_components[k] += tasks_components[k]
            tasks_train_batches += 1

        tasks_train_avg = {k: v / tasks_train_batches for k, v in tasks_train_components.items()}

        # ── Task validate ──────────────────────────────────────────
        tasks_val_components = {"loss": 0, "reg": 0, "cls": 0}
        tasks_val_batches    = 0
        total_acc            = 0.0
        total_mae            = np.zeros(3)

        for batch in val_loader:
            tasks_components, accuracy, mae = validation_step_tasks(model, batch, device)
            for k in tasks_val_components:
                tasks_val_components[k] += tasks_components[k]
            tasks_val_batches += 1
            total_acc         += accuracy
            total_mae         += mae


        tasks_val_avg = {k: v / tasks_val_batches for k, v in tasks_val_components.items()}
        acc_avg       = total_acc / tasks_val_batches
        mae_avg       = total_mae / tasks_val_batches

        scheduler.step(tasks_val_avg["loss"])

        # ── Update history ─────────────────────────────────────────
        track_losses["train_reg"].append(tasks_train_avg["reg"])
        track_losses["train_cls"].append(tasks_train_avg["cls"])
        track_losses["val_reg"].append(tasks_val_avg["reg"])
        track_losses["val_cls"].append(tasks_val_avg["cls"])
        track_losses["val_acc"].append(acc_avg)
        track_losses["val_mae"].append(mae_avg)

        # ── Log to Comet ───────────────────────────────────────────
        exp.log_metrics({
            "Train/reg"        : tasks_train_avg["reg"],
            "Train/cls"        : tasks_train_avg["cls"],
            "Validation/reg"   : tasks_val_avg["reg"],
            "Validation/cls"   : tasks_val_avg["cls"],
            "Validation/acc"   : acc_avg,
            "Validation/mae_teff" : mae_avg[0],
            "Validation/mae_logg" : mae_avg[1],
            "Validation/mae_feh"  : mae_avg[2],
        }, epoch= n_epochs + epoch)

    exp.end()
    return exp, track_losses

def main():
    parser = argparse.ArgumentParser(prog="Trainer_diagnose")
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--lr',  type=float, default=0.0003)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument("--batch_size", type= int, default= 256)
    
    args = parser.parse_args()

    train_loader, val_loader, test_loader = prepare_dataloader(
    flux      = flux_clean,
    y_reg     = y_reg_norm,
    y_cls     = y_cls_clean,
    batch_size = args.batch_size
    )

    model = DownHeads(
    input_dim   = flux_clean.shape[1],
    latent_dim  = 128,
    num_classes = 7
)
    
    train(model, train_loader, val_loader, batch_size=args.batch_size, n_epochs=args.max_epochs, lr = args.lr, beta = args.beta)


if __name__ == "__main__":
    main()




        


    




