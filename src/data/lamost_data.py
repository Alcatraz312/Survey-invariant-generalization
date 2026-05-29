import os
import gc 
import numpy as np
import h5py
from tqdm import tqdm
import pandas as pd

# load SDSS training statistics for label normalisation
sdss_reg_mu    = np.load("/home/arbiter/projects/Survey-invariant-generalization/src/models/reg_mu.npy")      # SDSS mean
sdss_reg_sigma = np.load("/home/arbiter/projects/Survey-invariant-generalization/src/models/reg_sigma.npy")   # SDSS std

def get_flux_labels(data, meta_data):
    '''
    Extract flux arrays and corresponding labels from the HDF5 file.
    Looks up each star in the metadata CSV by obsid and retrieves
    atmospheric parameters and spectral class.
    Returns flux array (N, 3800), regression labels (N, 3), class indices (N,).
    '''
    
    flux_list = []
    teff_list = []
    logg_list = []
    feh_list  = []
    cls_list  = []
    failed    = 0

    # integer mapping from MK class letter to class index
    class_map = {"O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6}

    for star_id in tqdm(data):
        try:
            # load resampled flux array for this star
            flux_array = data[star_id]["flux"][:]

            # convert group key to integer obsid for metadata lookup
            obsid = int(star_id)

            # find matching row in metadata CSV
            star_meta = meta_data.loc[meta_data["obsid"] == obsid]

            # skip if no metadata match found
            if len(star_meta) == 0:
                failed += 1
                continue

            # extract atmospheric parameters
            teff = star_meta["teff"].iloc[0]
            logg = star_meta["logg"].iloc[0]
            feh  = star_meta["feh"].iloc[0]

            # take first character of subclass string as MK class letter
            spectral_class = star_meta["subclass"].iloc[0][0]

            # skip stars with missing parameter labels
            if pd.isna(teff) or pd.isna(logg) or pd.isna(feh):
                failed += 1
                continue

            # skip stars whose class is not in the standard MK map
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

    # stack lists into arrays
    flux_array = np.array(flux_list)
    y_reg      = np.column_stack([teff_list, logg_list, feh_list])  # shape (N, 3)
    y_cls      = np.array(cls_list)                                  # shape (N,)

    return flux_array, y_reg, y_cls

def normalise_labels(y_reg):
    '''
    Standardise each atmospheric parameter to zero mean and unit variance.
    Computes mean and std from the input array and returns the normalised
    array along with the statistics needed for denormalisation later.
    Returns normalised array + stats for denormalisation later.
    '''
    mu    = y_reg.mean(axis=0)    # shape -> (3,)
    sigma = y_reg.std(axis=0)     # shape -> (3,)
    return (y_reg - mu) / sigma, mu, sigma


def denormalise_labels(y_norm, mu, sigma):
    '''
    Reverse the standardisation applied by normalise_labels.
    Converts normalised predictions back to physical units.
    '''
    return y_norm * sigma + mu

def clean_stars(flux, y_reg, y_cls):
    '''
    Remove low-quality spectra from the dataset using two filters.
    First filter: drops stars whose median normalised flux exceeds 5.0,
    indicating a failed flux calibration or normalisation.
    Second filter: drops stars containing any pixel outside [-3.0, 10.0],
    indicating cosmic ray hits or instrumental artifacts.
    Returns cleaned flux, regression labels, and class indices.
    '''

    # filter 1 — remove stars with abnormally high median flux
    per_star_median = np.median(flux, axis=1)
    good_norm_mask  = per_star_median < 5.0

    flux  = flux[good_norm_mask]
    y_reg = y_reg[good_norm_mask]
    y_cls = y_cls[good_norm_mask]

    print(f"After norm filter  — kept: {good_norm_mask.sum()} | dropped: {(~good_norm_mask).sum()}")

    # filter 2 — remove stars with outlier pixel values
    star_index_list = []
    for i in range(len(flux)):

        star = flux[i]
        # find pixels outside the acceptable range
        star = star[(star > 10) | (star < -3.0)]
        if len(star) > 0:
            star_index_list.append(i)   # mark this star as bad

    # build a boolean mask that excludes bad stars
    bad_indices  = np.array(star_index_list)
    quality_mask = np.ones(len(flux), dtype=bool)
    quality_mask[bad_indices] = False

    flux_clean  = flux[quality_mask]
    y_reg_clean = y_reg[quality_mask]
    y_cls_clean = y_cls[quality_mask]

    return flux_clean, y_reg_clean, y_cls_clean    

# load resampled LAMOST HDF5 file and metadata CSV
data_path = "/home/arbiter/projects/Survey-invariant-generalization/data/usable_data"
lamost_data = h5py.File(f"{data_path}/lamost_resampled.h5")
lamost_meta_data = pd.read_csv(f"{data_path}/lamost_meta.csv")

# extract flux and labels then apply quality cleaning
flux, y_reg, y_cls = get_flux_labels(lamost_data, lamost_meta_data)
lamost_flux_clean, y_reg_clean, lamost_y_cls_clean = clean_stars(flux, y_reg, y_cls)

# sanity check on flux range after cleaning
print(lamost_flux_clean.min())
print(lamost_flux_clean.max())

# normalise LAMOST labels using SDSS training statistics
# this keeps both surveys in the same label coordinate system
lamost_y_reg_norm = (y_reg_clean - sdss_reg_mu) / sdss_reg_sigma

# save outputs for use in training and evaluation
np.save("lamost_y_reg_raw.npy", y_reg_clean)       # raw physical label values
np.save("lamost_y_cls.npy", lamost_y_cls_clean)    # class indices for UMAP colouring
np.save("lamost_flux.npy", lamost_flux_clean)      # cleaned flux arrays