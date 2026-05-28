import os
import gc 
import numpy as np
import h5py
from tqdm import tqdm
import pandas as pd

def get_flux_labels(data, meta_data):
    
    ''' 
    Preparing the features and labels for the experiments \n
    flux and their corresponding spectral MK classes and atmospheric parameters 
    '''

    flux_list     = []
    teff_list     = []
    logg_list     = []
    feh_list      = []
    cls_list      = []
    failed        = 0

    # integer index for each MK spectral class
    class_map = {"O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6}

    for star_id in tqdm(data):
        # star_id is formatted as "plate-fiberid-mjd"
        star_id_info = star_id.split("-")

        try:
            # load flux array for this star from HDF5
            flux_array = data[star_id]["flux"][:]

            # look up the star's metadata using plate, fiberid, mjd
            star_meta = meta_data.loc[
                (meta_data["plate"]   == int(star_id_info[0])) &
                (meta_data["fiberid"] == int(star_id_info[1])) &
                (meta_data["mjd"]     == int(star_id_info[2]))
            ]

            # skip stars with no matching metadata entry
            if len(star_meta) == 0:
                failed += 1
                continue

            # retrieve atmospheric parameter labels from SSPP pipeline
            teff = star_meta["elodieTeff"].iloc[0]
            logg = star_meta["elodieLogG"].iloc[0]
            feh  = star_meta["elodieFeH"].iloc[0]

            # first character of subclass gives the broad MK class (e.g. "G" from "G2V")
            spectral_class = star_meta["subclass"].iloc[0][0]

            # skip if any label is missing
            if pd.isna(teff) or pd.isna(logg) or pd.isna(feh):
                failed += 1
                continue

            # skip if spectral class is not one of the seven standard MK classes
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

    # stack lists into arrays ready for training
    flux_array  = np.array(flux_list)                                # (n_stars, 3800)
    y_reg       = np.column_stack([teff_list, logg_list, feh_list])  # (n_stars, 3)
    y_cls       = np.array(cls_list)                                  # (n_stars,)

    return flux_array, y_reg, y_cls


def normalise_labels(y_reg):
    '''
    Standardise each parameter to zero mean unit variance.
    Returns normalised array + stats for denormalisation later.
    '''
    mu    = y_reg.mean(axis=0)    # per-parameter mean,  shape (3,)
    sigma = y_reg.std(axis=0)     # per-parameter std,   shape (3,)
    return (y_reg - mu) / sigma, mu, sigma


def denormalise_labels(y_norm, mu, sigma):
    # reverse the standardisation to recover physical units
    return y_norm * sigma + mu


def clean_stars(flux, y_reg, y_cls):

    # remove stars whose median normalised flux exceeds 5.0
    # a median this high indicates a flux calibration failure
    per_star_median = np.median(flux, axis=1)
    good_norm_mask  = per_star_median < 5.0

    flux  = flux[good_norm_mask]
    y_reg = y_reg[good_norm_mask]
    y_cls = y_cls[good_norm_mask]

    print(f"After norm filter  — kept: {good_norm_mask.sum()} | dropped: {(~good_norm_mask).sum()}")

    # find stars that contain at least one pixel outside the range [-3, 10]
    # these indicate cosmic ray hits or bad columns
    star_index_list = []
    for i in range(len(flux)):
        star = flux[i]
        star = star[(star > 10) | (star < -3.0)]
        if len(star) > 0:
            star_index_list.append(i)

    # build a boolean mask that excludes the bad stars found above
    bad_indices  = np.array(star_index_list)
    quality_mask = np.ones(len(flux), dtype=bool)
    quality_mask[bad_indices] = False

    flux_clean  = flux[quality_mask]
    y_reg_clean = y_reg[quality_mask]
    y_cls_clean = y_cls[quality_mask]

    return flux_clean, y_reg_clean, y_cls_clean    


# load resampled SDSS spectra and metadata from disk
data_path = "/home/arbiter/projects/Survey-invariant-generalization/data/usable_data"
sdss_data      = h5py.File(f"{data_path}/sdss_resampled.h5")
sdss_meta_data = pd.read_csv(f"{data_path}/sdss_meta.csv")

# extract flux arrays and labels, then apply quality cleaning
flux, y_reg, y_cls = get_flux_labels(sdss_data, sdss_meta_data)
flux_clean, y_reg_clean, y_cls_clean = clean_stars(flux, y_reg, y_cls)

# sanity check on flux range after cleaning
print(flux_clean.min())
print(flux_clean.max())

# standardise labels and save normalisation statistics
# these stats are reused for LAMOST normalisation to keep coordinate systems consistent
y_reg_norm, reg_mu, reg_sigma = normalise_labels(y_reg_clean)

np.save("reg_mu.npy", reg_mu)       # (3,)  per-parameter mean
np.save("reg_sigma.npy", reg_sigma) # (3,)  per-parameter standard deviation