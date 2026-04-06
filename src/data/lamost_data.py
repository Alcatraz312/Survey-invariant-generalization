import os
import gc 
import numpy as np
import h5py
from tqdm import tqdm
import pandas as pd

sdss_reg_mu    = np.load("/home/arbiter/projects/Survey-invariant-generalization/src/models/reg_mu.npy")      # SDSS mean
sdss_reg_sigma = np.load("/home/arbiter/projects/Survey-invariant-generalization/src/models/reg_sigma.npy")   # SDSS std

def get_flux_labels(data, meta_data):
    
    flux_list = []
    teff_list = []
    logg_list = []
    feh_list  = []
    cls_list  = []
    failed    = 0

    class_map = {"O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6}

    for star_id in tqdm(data):
        try:
            # Flux
            flux_array = data[star_id]["flux"][:]

            # Strip leading '1' to match obsid in metadata
            obsid = int(star_id[1:])

            # Metadata lookup
            star_meta = meta_data.loc[meta_data["obsid"] == obsid]

            # Skip if no match found
            if len(star_meta) == 0:
                failed += 1
                continue

            teff = star_meta["teff"].iloc[0]
            logg = star_meta["logg"].iloc[0]
            feh  = star_meta["feh"].iloc[0]
            spectral_class = star_meta["subclass"].iloc[0][0]

            # Skip if any label is null
            if pd.isna(teff) or pd.isna(logg) or pd.isna(feh):
                failed += 1
                continue

            # Skip if class not in map
            if spectral_class not in class_map:
                failed += 1
                continue

            flux_list.append(flux_array)
            teff_list.append(teff)
            logg_list.append(logg)
            feh_list.append(feh)
            cls_list.append(class_map[spectral_class])

        except Exception as e:
            print(f"Failed for {star_id}: {e}")
            failed += 1
            continue

    print(f"Collected: {len(flux_list)} | Failed/skipped: {failed}")

    flux_array = np.array(flux_list)
    y_reg      = np.column_stack([teff_list, logg_list, feh_list])
    y_cls      = np.array(cls_list)

    return flux_array, y_reg, y_cls

def normalise_labels(y_reg):
    '''
    Standardise each parameter to zero mean unit variance.
    Returns normalised array + stats for denormalisation later.
    '''
    mu    = y_reg.mean(axis=0)    # shape -> (3,)
    sigma = y_reg.std(axis=0)     # shape -> (3,)
    return (y_reg - mu) / sigma, mu, sigma


def denormalise_labels(y_norm, mu, sigma):
    return y_norm * sigma + mu

def clean_stars(flux, y_reg, y_cls):

    per_star_median = np.median(flux, axis=1)
    good_norm_mask  = per_star_median < 5.0

    flux  = flux[good_norm_mask]
    y_reg = y_reg[good_norm_mask]
    y_cls = y_cls[good_norm_mask]

    print(f"After norm filter  — kept: {good_norm_mask.sum()} | dropped: {(~good_norm_mask).sum()}")

    star_index_list = []
    for i in range(len(flux)):

        star = flux[i]
        star = star[(star > 10) | (star < -3.0)]
        if len(star) > 0:
            star_index_list.append(i)

    bad_indices  = np.array(star_index_list)
    quality_mask = np.ones(len(flux), dtype=bool)
    quality_mask[bad_indices] = False

    flux_clean  = flux[quality_mask]
    y_reg_clean = y_reg[quality_mask]
    y_cls_clean = y_cls[quality_mask]

    return flux_clean, y_reg_clean, y_cls_clean    

data_path = "/home/arbiter/projects/Survey-invariant-generalization/data/usable_data"
lamost_data = h5py.File(f"{data_path}/lamost_resampled.h5")
lamost_meta_data = pd.read_csv(f"{data_path}/lamost_meta.csv")

flux, y_reg, y_cls = get_flux_labels(lamost_data, lamost_meta_data)
flux_clean, y_reg_clean, y_cls_clean = clean_stars(flux, y_reg, y_cls)

print(flux_clean.min())
print(flux_clean.max())

y_reg_norm = (y_reg_clean - sdss_reg_mu) / sdss_reg_sigma

np.save("lamost_y_reg_raw.npy", y_reg_clean)  # save raw physical values
np.save("lamost_y_cls.npy", y_cls_clean)      # for H1 UMAP coloring
np.save("lamost_flux.npy", flux_clean)         # might need later