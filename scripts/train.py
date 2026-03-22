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

comet_key = os.environ.get("comet_ml_key")


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

def get_beta(epoch, warmup_epochs=30, beta_max=0.5):
    if epoch < warmup_epochs:
        return beta_max * (epoch / warmup_epochs)
    return beta_max

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

    exp = comet_ml.start(api_key = comet_key, project_name = "Dissertation ML")
    exp.set_name(f"Multitasking Experiment Batch Size :{batch_size} , lr_tasks_encoder : {lr_tasks}, lr_reconstruction : {lr_recon} , epochs : {n_epochs} , loss : {loss_agg}")

    # log hyperparameters into comet ML

    exp.log_parameters({
        "n_epochs"   : n_epochs,
        "lr_tasks_encoder"         : lr_tasks,
        "lr_recon" : lr_recon,
        "beta"       : beta,
        "latent_dim" : model.mu_layer.out_features,
        "input_dim"  : model.encoder[0].in_features,
        "batch_size" : train_loader.batch_size,
        "Loss_Aggregation" : "Linear Sum" if loss_agg == "s" else "Uncertainty Weighting loss aggregation",
        "LR scheduler patience" : lr_patience,
        "LR min" : lr_min,
        "Schedule LR ?" : schedule_lr
    })

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

        # logging the changing learning rates onto comet 
        task_lr = optimizer.param_groups[0]["lr"]
        recon_lr = optimizer.param_groups[1]["lr"]
        exp.log_metric("lr/tasks", task_lr, epoch = epoch)
        exp.log_metric("lr/reconstruction", recon_lr, epoch = epoch)

        # update loss tracks

        track_losses["train_loss"].append(train_avg["loss"])
        track_losses["train_recon"].append(train_avg["recon"])
        track_losses["train_kl"].append(train_avg["kl"])
        track_losses["train_cls"].append(train_avg["cls"])
        track_losses["train_reg"].append(train_avg["reg"])

        track_losses["val_loss"].append(val_avg["loss"])
        track_losses["val_recon"].append(val_avg["recon"])
        track_losses["val_kl"].append(val_avg["kl"])
        track_losses["val_cls"].append(val_avg["cls"])
        track_losses["val_reg"].append(val_avg["reg"])
        track_losses["val_acc"].append(acc_avg)
        track_losses["val_mae"].append(mae_avg)
        
        # log the experiment loss metrics and parameters to comet ml 
        if loss_agg == "u":
            exp.log_metrics({
                # Train losses
                "train/loss"  : train_avg["loss"],
                "train/recon" : train_avg["recon"],
                "train/kl"    : train_avg["kl"],
                "train/reg"   : train_avg["reg"],
                "train/cls"   : train_avg["cls"],
                # Val losses
                "val/loss"    : val_avg["loss"],
                "val/recon"   : val_avg["recon"],
                "val/kl"      : val_avg["kl"],
                "val/reg"     : val_avg["reg"],
                "val/cls"     : val_avg["cls"],
                # Val metrics
                "val/accuracy"     : acc_avg,
                "val/mae_teff"     : mae_avg[0],
                "val/mae_logg"     : mae_avg[1],
                "val/mae_feh"      : mae_avg[2],
                # Learned task uncertainties
                "sigma/recon" : torch.exp(0.5 * model.log_sigma_recon).item(),
                "sigma/reg"   : torch.exp(0.5 * model.logvar_reg).item(),
                "sigma/cls"   : torch.exp(0.5 * model.logvar_cls).item(),
            }, epoch=epoch)

        elif loss_agg == "s":
            exp.log_metrics({
                # Train losses
                "train/loss"  : train_avg["loss"],
                "train/recon" : train_avg["recon"],
                "train/kl"    : train_avg["kl"],
                "train/reg"   : train_avg["reg"],
                "train/cls"   : train_avg["cls"],
                # Val losses
                "val/loss"    : val_avg["loss"],
                "val/recon"   : val_avg["recon"],
                "val/kl"      : val_avg["kl"],
                "val/reg"     : val_avg["reg"],
                "val/cls"     : val_avg["cls"],
                # Val metrics
                "val/accuracy"     : acc_avg,
                "val/mae_teff"     : mae_avg[0],
                "val/mae_logg"     : mae_avg[1],
                "val/mae_feh"      : mae_avg[2],
            }, epoch=epoch)

        # checking to see the parameters 
        # counter += 1
        # if counter < 2:
        #     print(optimizer.param_groups[1])

    exp.end()

    return exp

# constant seed to run experiments
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(prog="Trainer")
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--lr_tasks',  type=float, default=0.0003)
    parser.add_argument("--lr_recon", type = float, default = 0.0001)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument("--batch_size", type= int, default= 256 )
    parser.add_argument("--loss_agg", type = str, default = "s")
    parser.add_argument("--lr_patience", type = int, default= 10)
    parser.add_argument("--lr_min", type = float, default = 1e-5)
    parser.add_argument("--schedule_lr", type= bool, default= True)
    parser.add_argument("--seed", type = int, default = 42)
    args = parser.parse_args()

    set_seed(args.seed)

    # lr_list = [1e-3, 1e-4, 3e-4, 1e-5]
    
    train_loader, val_loader, test_loader = prepare_dataloader(
    flux      = flux_clean,
    y_reg     = y_reg_norm,
    y_cls     = y_cls_clean,
    batch_size = args.batch_size
    )

    model = MTLArchitecture(
        input_dim   = flux_clean.shape[1],
        latent_dim  = 128,
        num_classes = 7
    )
    train(model=model, train_loader=train_loader, val_loader=val_loader, batch_size= args.batch_size,
            loss_agg= args.loss_agg, n_epochs=args.max_epochs, lr_tasks= args.lr_tasks, lr_recon=args.lr_recon, beta=args.beta, lr_min = args.lr_min, schedule_lr= args.schedule_lr)

if __name__ == '__main__':
    main()