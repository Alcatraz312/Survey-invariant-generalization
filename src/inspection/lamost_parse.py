import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt 
from astropy.io import fits

import json

folder_directory = "/home/arbiter/projects/data_downloads"

folder_list = os.listdir(folder_directory)
folder_list = [x for x in folder_list if x != "Lamost_metadata"]

folder_list.sort(key=lambda x: int(x.split("_")[-1]))

directory_list = []

for folder in folder_list:
    for root, dirs, files in os.walk(f"{folder_directory}/{folder}"):
        root = root.split("/")
        if len(root) == 8:
            directory_list.append("/".join(root))


metadata_lamost_dir = f"{folder_directory}/Lamost_metadata"
metadata_file_list = os.listdir(metadata_lamost_dir)

metadata_file_list.sort(key=lambda x: int(x.split("_")[1]))

metadataframe_list = []
for file in metadata_file_list:
    if "Zone" not in file:
        dataframe = pd.read_csv(f"{metadata_lamost_dir}/{file}")

        metadataframe_list.append(dataframe)

meta_df = pd.concat(metadataframe_list, axis = 0)

import os
import numpy as np
from astropy.io import fits

def data_consol(directory_list, meta_df):
    lamost_data_dict = {}
    global_idx = 0

    for directory in directory_list:
        file_list = os.listdir(directory)

        for file in file_list:
            # only valid LAMOST spectra
            if "fits.gz" not in file or ":Zone" in file:
                continue

            with fits.open(os.path.join(directory, file)) as stellar_data:
                primary_hdu = stellar_data[0]
                secondary_data = stellar_data[1].data

                # spectral data
                flux = secondary_data["FLUX"][0]
                wavelength_grid = secondary_data["WAVELENGTH"][0]

                # OBSID from FITS header
                obsid = primary_hdu.header["OBSID"]

                # metadata lookup (SAFE)
                row = meta_df.loc[meta_df["obsid"] == obsid]
                if row.empty:
                    continue  # skip stars without metadata

                # extract scalar values
                obs_id = int(row["obsid"].iloc[0])
                teff = float(row["teff"].iloc[0])
                logg = float(row["logg"].iloc[0])
                feh = float(row["feh"].iloc[0])
                spectral_class = str(row["subclass"].iloc[0])

                # SNR values
                snru = float(primary_hdu.header.get("SNRU", np.nan))
                snrg = float(primary_hdu.header.get("SNRG", np.nan))
                snrr = float(primary_hdu.header.get("SNRR", np.nan))
                snri = float(primary_hdu.header.get("SNRI", np.nan))
                snrz = float(primary_hdu.header.get("SNRZ", np.nan))

                # store star data
                lamost_data_dict[global_idx] = {
                    "flux": flux,
                    "wavelength_grid": wavelength_grid,
                    "obsID": obs_id,
                    "teff": teff,
                    "logg": logg,
                    "feh": feh,
                    "spectral_class": spectral_class,
                    "SNRU": snru,
                    "SNRG": snrg,
                    "SNRR": snrr,
                    "SNRI": snri,
                    "SNRZ": snrz
                }

                global_idx += 1

    return lamost_data_dict

                
lamost_data_dict = data_consol(directory_list = directory_list, meta_df= meta_df) 

# consolidate to Hdf5

