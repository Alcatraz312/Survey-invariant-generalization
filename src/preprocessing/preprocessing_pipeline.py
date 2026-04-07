import numpy as np
import pandas as pd

import h5py
import spectres

from tqdm import tqdm


sdss_data = h5py.File("/home/arbiter/projects/Survey-invariant-generalization/data/sdss_final_ready.h5")

def normalize_flux(flux, reference_grid, lambda_ref):
    ref_idx = np.argmin(np.abs(reference_grid - lambda_ref))
    ref_flux = flux[ref_idx]

    if ref_flux <= 0:
        return None   # bad spectrum — negative or zero at reference point

    return flux / ref_flux


def mask_resample(star_data):

    failed_ids = 0

    LAMBDA_START = 4000
    LAMBDA_END = 8500
    NUM_PIXELS = 3800

    reference_grid = np.linspace(LAMBDA_START, LAMBDA_END, NUM_PIXELS)

    results_dict = {}
    counter = 0
    for star in tqdm(star_data):
    
        star_wave = star_data[star]["wavelength_grid"][:]
        star_flux = star_data[star]["flux"][:]
        star_snr = star_data[star]["snr"][:]

        mask  = (star_wave >= LAMBDA_START) & (star_wave <= LAMBDA_END)

        star_wave_masked = star_wave[mask]
        star_flux_masked = star_flux[mask]
        star_snr_masked = star_snr[mask]

        if len(star_wave_masked) < 3000:
            failed_ids += 1
            continue


        try:
            flux_resampled = spectres.spectres(
                    reference_grid,
                    star_wave_masked,
                    star_flux_masked,
                    fill = 0.0,
                    verbose = False
                )
            
            snr_resampled = np.interp(reference_grid, star_wave_masked, star_snr_masked)

            normalized_resampled_flux = normalize_flux(flux_resampled, reference_grid, 5550.0)

        except Exception as e:
            print(f"Failed star {star}")
            continue
        

        results_dict[star] = {"wavelength_grid" : reference_grid,
                              "flux" : normalized_resampled_flux,
                              "snr" : snr_resampled}


    with h5py.File("/home/arbiter/projects/Survey-invariant-generalization/data/sdss_resampled.h5", "w") as f:

         for key, value in results_dict.items():
             grp = f.create_group(key)

             grp.create_dataset("wavelength_grid", data=value["wavelength_grid"])
             grp.create_dataset("flux", data=value["flux"])
             grp.create_dataset("snr", data=value["snr"])

    print(failed_ids)

mask_resample(sdss_data)