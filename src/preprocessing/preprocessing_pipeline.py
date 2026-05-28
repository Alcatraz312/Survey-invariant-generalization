import numpy as np
import pandas as pd

import h5py
import spectres

from tqdm import tqdm


sdss_data = h5py.File("/home/arbiter/projects/Survey-invariant-generalization/data/sdss_final_ready.h5")

def normalize_flux(flux, reference_grid, lambda_ref):      # normalizing flux at a reference wavelength pixel
    ref_idx = np.argmin(np.abs(reference_grid - lambda_ref))
    ref_flux = flux[ref_idx]

    if ref_flux <= 0:
        return None   # bad spectrum — negative or zero at reference point

    return flux / ref_flux


def mask_resample(star_data):               # flux resampling using spectres

    failed_ids = 0

    LAMBDA_START = 4000       # reference grid start
    LAMBDA_END = 8500           # reference grid end 
    NUM_PIXELS = 3800        # number of wavelength pixels

    reference_grid = np.linspace(LAMBDA_START, LAMBDA_END, NUM_PIXELS)              

    results_dict = {}

    for star in tqdm(star_data):
    
        star_wave = star_data[star]["wavelength_grid"][:]
        star_flux = star_data[star]["flux"][:]
        star_snr = star_data[star]["snr"][:]

        mask  = (star_wave >= LAMBDA_START) & (star_wave <= LAMBDA_END)     # mask

        # masking flux, wavelength and snr arrays
        star_wave_masked = star_wave[mask]
        star_flux_masked = star_flux[mask]
        star_snr_masked = star_snr[mask]

        # star failed if flux array has less than 3000 points after the mask
        if len(star_wave_masked) < 3000:     
            failed_ids += 1
            continue

        # spectral resampling using spectres        
        try:
            flux_resampled = spectres.spectres(
                    reference_grid,
                    star_wave_masked,
                    star_flux_masked,
                    fill = 0.0,
                    verbose = False
                )
            
            snr_resampled = np.interp(reference_grid, star_wave_masked, star_snr_masked)        # interpolating the SNR because it cannot be resampled 

            normalized_resampled_flux = normalize_flux(flux_resampled, reference_grid, 5550.0)

        except Exception as e:    # printing failed stars
            print(f"Failed star {star}")
            continue
        

        results_dict[star] = {"wavelength_grid" : reference_grid,
                              "flux" : normalized_resampled_flux,
                              "snr" : snr_resampled}

    # store cleaned stars in hdf5 format
    with h5py.File("/home/arbiter/projects/Survey-invariant-generalization/data/sdss_resampled.h5", "w") as f:

         for key, value in results_dict.items():
             grp = f.create_group(key)

             grp.create_dataset("wavelength_grid", data=value["wavelength_grid"])
             grp.create_dataset("flux", data=value["flux"])
             grp.create_dataset("snr", data=value["snr"])

    print(failed_ids)

mask_resample(sdss_data)