import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import json
import os

# load SDSS training statistics for denormalising predictions back to physical units
reg_mu    = np.load("/home/arbiter/projects/Survey-invariant-generalization/src/models/reg_mu.npy")
reg_sigma = np.load("/home/arbiter/projects/Survey-invariant-generalization/src/models/reg_sigma.npy")

# load LAMOST flux arrays and normalised ground truth labels
lamost_flux = np.load("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/lamost_flux.npy")
lamost_true = np.load("/home/arbiter/projects/Survey-invariant-generalization/src/inspection/model_inspection/lamost_true_reg.npy")

# parameter names for printing and saving results
param_names = ["T_eff (K)", "log g (dex)", "[Fe/H] (dex)"]

# create output directory if it does not exist
save_dir = "/home/arbiter/projects/Survey-invariant-generalization/experiments/experiments_data"
os.makedirs(save_dir, exist_ok=True)


# BASELINE 1 — Mean predictor
# predicts the SDSS training mean (zero in normalised space) for every star
# represents the floor — any useful model should beat this

print("=" * 20 + " Baseline Mean " + "=" * 20)

# zero array represents predicting the SDSS mean for all stars
mean_pred    = np.zeros_like(lamost_true)
mae_mean     = []
mae_mean_dict = {}

for i, (name, sigma) in enumerate(zip(param_names, reg_sigma)):
    # multiply by sigma to convert MAE from normalised space to physical units
    mae = np.mean(np.abs(mean_pred[:, i] - lamost_true[:, i])) * sigma
    mae_mean.append(mae)
    mae_mean_dict[name] = float(mae)
    print(f"{name}: {mae:.4f}")

# BASELINE 2 — Linear Ridge regression on raw flux
# trains directly on LAMOST flux — a strong in-domain baseline
# the VAE never sees LAMOST during training, ridge does

print("\n" + "=" * 20 + " Baseline Ridge Regression " + "=" * 20)

# 70/30 split — ridge trains on 70%, evaluated on 30%
# idx_test saved so VAE can be evaluated on the exact same subset
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    lamost_flux, lamost_true, np.arange(len(lamost_true)),
    test_size=0.3, random_state=42
)

# fit ridge regression with L2 regularisation strength alpha=1.0
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)

mae_ridge     = []
mae_ridge_dict = {}

for i, (name, sigma) in enumerate(zip(param_names, reg_sigma)):
    # convert MAE back to physical units using SDSS sigma
    mae = np.mean(np.abs(ridge_preds[:, i] - y_test[:, i])) * sigma
    mae_ridge.append(mae)
    mae_ridge_dict[name] = float(mae)
    print(f"MAE {name}: {mae:.4f}")

# save test indices so VAE can be evaluated on same subset
np.save(f"{save_dir}/baseline_idx_test.npy", idx_test)

# save baseline MAE results as JSON files for use in H3 hypothesis testing
with open(f"{save_dir}/mean_mae_reg.json", "w") as f:
    json.dump(mae_mean_dict, f, indent=2)

with open(f"{save_dir}/ridge_mae_reg.json", "w") as f:
    json.dump(mae_ridge_dict, f, indent=2)

print("\nBaseline results saved.")