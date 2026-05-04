from src.models.multitask import MTLArchitecture, smart_init
import argparse
from src.data.data import flux_clean, y_reg_norm, y_cls_clean
from src.data.lamost_data import lamost_flux_clean, lamost_y_reg_norm, lamost_y_cls_clean

from experiments.sdss_dataloader import prepare_dataloader
from experiments.lamost_dataloader import prepare_dataloader_lamost
from experiments.model_train import train, set_seed

import torch
import torch.nn as nn   
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, random_split

import os
import numpy as np

import gc


def main():
    parser = argparse.ArgumentParser(prog="Trainer")
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--lr_tasks',  type=float, default=0.0003)
    parser.add_argument("--lr_recon", type = float, default = 0.0001)
    parser.add_argument('--beta', type=float, default=0.03)
    parser.add_argument("--batch_size", type= int, default= 256)
    parser.add_argument("--loss_agg", type = str, default = "u")
    parser.add_argument("--lr_patience", type = int, default= 10)
    parser.add_argument("--lr_min", type = float, default = 1e-5)
    parser.add_argument("--schedule_lr", type= bool, default= True)
    parser.add_argument("--seed", type = int, default = 123)
    parser.add_argument("--latent_dim", type = int, default= 128)
    args = parser.parse_args()

    set_seed(args.seed) 

    train_loader, val_loader, test_loader = prepare_dataloader(
    flux      = flux_clean,
    y_reg     = y_reg_norm,
    y_cls     = y_cls_clean,
    batch_size = args.batch_size
    )

    model = MTLArchitecture(
        input_dim   = flux_clean.shape[1],
        latent_dim  = args.latent_dim,
        num_classes = 7
    )

    smart_init(model= model)

    train(model=model, train_loader=train_loader, val_loader=val_loader, batch_size= args.batch_size,
            loss_agg= args.loss_agg, n_epochs=args.max_epochs, lr_tasks= args.lr_tasks, lr_recon=args.lr_recon, beta=args.beta, lr_min = args.lr_min, schedule_lr= args.schedule_lr)
    
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # transfer to cuda cores

    model = model.to(device)

    checkpoint = torch.load("/home/arbiter/projects/Survey-invariant-generalization/src/models/final_model.pth")
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

    dir_created = f"/home/arbiter/projects/Survey-invariant-generalization/experiments/experiments_data/experiment_{args.seed}"

    os.makedirs(dir_created, exist_ok=True)

    np.save(f"{dir_created}/sdss_preds_reg.npy", preds_reg)
    np.save(f"{dir_created}/sdss_true_reg.npy", true_reg)
    np.save(f"{dir_created}/sdss_preds_cls.npy", preds_cls)
    np.save(f"{dir_created}/sdss_true_cls.npy", true_cls)
    np.save(f"{dir_created}/sdss_latent_mu.npy", all_mu)
    np.save(f"{dir_created}/sdss_reconstructed_spectra.npy", all_x_hat)
    np.save(f"{dir_created}/sdss_flux.npy", all_flux)

    gc.collect()

    lamost_loader = prepare_dataloader_lamost(
        flux = lamost_flux_clean, 
        y_reg = lamost_y_reg_norm,
        y_cls = lamost_y_cls_clean,
        batch_size= 256
    )

    all_preds_reg = []
    all_true_reg  = []
    all_preds_cls = []
    all_true_cls  = []
    all_mu = []
    all_x_hat = []
    all_flux = []

    with torch.no_grad():
        for batch in lamost_loader:
            flux, y_reg, y_cls = [b.to(device) for b in batch]
            
            out = model(flux)
            all_preds_reg.append(out["mu_reg"].cpu().numpy())
            all_true_reg.append(y_reg.cpu().numpy())
            all_mu.append(out["mu"].cpu().numpy())

            all_x_hat.append(out["x_hat"].cpu().numpy())
            all_flux.append(flux.cpu().numpy())

            cls_preds = out["cls_logits"].argmax(dim=1)
            all_preds_cls.append(cls_preds.cpu().numpy())
            all_true_cls.append(y_cls.cpu().numpy())

    # concatenate all batches
    preds_reg = np.concatenate(all_preds_reg, axis=0)  # shape (N, 3)
    true_reg  = np.concatenate(all_true_reg,  axis=0)
    preds_cls = np.concatenate(all_preds_cls, axis=0)
    true_cls  = np.concatenate(all_true_cls,  axis=0)
    all_mu = np.concatenate(all_mu, axis=0) 
    all_x_hat = np.concatenate(all_x_hat, axis = 0)
    all_flux = np.concatenate(all_flux, axis = 0)

    np.save(f"{dir_created}/lamost_preds_reg.npy", preds_reg)
    np.save(f"{dir_created}/lamost_true_reg.npy", true_reg)
    np.save(f"{dir_created}/lamost_preds_cls.npy", preds_cls)
    np.save(f"{dir_created}/lamost_true_cls.npy", true_cls)
    np.save(f"{dir_created}/lamost_latent_mu.npy", all_mu)
    np.save(f"{dir_created}/lamost_reconstructed_spectra.npy", all_x_hat)
    np.save(f"{dir_created}/lamost_flux.npy", all_flux)

if __name__ == '__main__':
    main()