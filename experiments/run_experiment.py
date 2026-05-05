from src.models.multitask import MTLArchitecture, smart_init
import argparse
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
    parser.add_argument('--max_epochs',  type=int,   default=200)
    parser.add_argument('--lr_tasks',    type=float, default=0.0003)
    parser.add_argument("--lr_recon",    type=float, default=0.0001)
    parser.add_argument('--beta',        type=float, default=0.03)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--loss_agg",    type=str,   default="u")
    parser.add_argument("--lr_patience", type=int,   default=10)
    parser.add_argument("--lr_min",      type=float, default=1e-5)
    parser.add_argument("--schedule_lr", type=bool,  default=True)
    parser.add_argument("--seed",        type=int,   default=1)
    parser.add_argument("--latent_dim",  type=int,   default=128)
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Step 1: Load SDSS and build dataloaders ──
    from src.data.data import flux_clean, y_reg_norm, y_cls_clean

    input_dim = flux_clean.shape[1]

    train_loader, val_loader, test_loader = prepare_dataloader(
        flux       = flux_clean,
        y_reg      = y_reg_norm,
        y_cls      = y_cls_clean,
        batch_size = args.batch_size
    )

    # Free raw SDSS arrays — dataloader holds references internally
    del flux_clean, y_reg_norm, y_cls_clean
    gc.collect()

    # ── Step 2: Build and train model ──
    model = MTLArchitecture(
        input_dim   = input_dim,
        latent_dim  = args.latent_dim,
        num_classes = 7
    )
    smart_init(model=model)

    train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        batch_size   = args.batch_size,
        loss_agg     = args.loss_agg,
        n_epochs     = args.max_epochs,
        lr_tasks     = args.lr_tasks,
        lr_recon     = args.lr_recon,
        beta         = args.beta,
        lr_min       = args.lr_min,
        schedule_lr  = args.schedule_lr,
        seed         = args.seed
    )

    # ── Step 3: SDSS inference (model already trained, no reload needed) ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dir_created = f"/home/arbiter/projects/Survey-invariant-generalization/experiments/experiments_data/experiment_{args.seed}"
    os.makedirs(dir_created, exist_ok=True)

    def run_inference(loader, device):
        all_preds_reg, all_true_reg  = [], []
        all_preds_cls, all_true_cls  = [], []
        all_mu, all_x_hat, all_flux  = [], [], []

        with torch.no_grad():
            for batch in loader:
                flux, y_reg, y_cls = [b.to(device) for b in batch]
                out = model(flux)

                all_preds_reg.append(out["mu_reg"].cpu().numpy())
                all_true_reg.append(y_reg.cpu().numpy())
                all_mu.append(out["mu"].cpu().numpy())
                all_x_hat.append(out["x_hat"].cpu().numpy())
                all_flux.append(flux.cpu().numpy())
                all_preds_cls.append(out["cls_logits"].argmax(dim=1).cpu().numpy())
                all_true_cls.append(y_cls.cpu().numpy())

        return {
            "preds_reg" : np.concatenate(all_preds_reg),
            "true_reg"  : np.concatenate(all_true_reg),
            "preds_cls" : np.concatenate(all_preds_cls),
            "true_cls"  : np.concatenate(all_true_cls),
            "latent_mu" : np.concatenate(all_mu),
            "x_hat"     : np.concatenate(all_x_hat),
            "flux"      : np.concatenate(all_flux),
        }

    # SDSS inference
    sdss_results = run_inference(test_loader, device)
    for key, val in sdss_results.items():
        np.save(f"{dir_created}/sdss_{key}.npy", val)

    # Free SDSS results and GPU cache
    del sdss_results, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ── Step 4: Load LAMOST and run inference ──
    from src.data.lamost_data import lamost_flux_clean, lamost_y_reg_norm, lamost_y_cls_clean

    lamost_loader = prepare_dataloader_lamost(
        flux      = lamost_flux_clean,
        y_reg     = lamost_y_reg_norm,
        y_cls     = lamost_y_cls_clean,
        batch_size = 256
    )

    del lamost_flux_clean, lamost_y_reg_norm, lamost_y_cls_clean
    gc.collect()

    lamost_results = run_inference(lamost_loader, device)
    for key, val in lamost_results.items():
        np.save(f"{dir_created}/lamost_{key}.npy", val)

    del lamost_results
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Seed {args.seed} complete — saved to {dir_created}")

if __name__ == '__main__':
    main()

    # Force cleanup on exit
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
