# Survey-Invariant Latent Representation of Stellar Spectra <img width="40" height="40" alt="telescope" src="https://github.com/user-attachments/assets/3d65f669-5728-4fec-b153-37fdf7c07167" />

<p align = "center">
<img src = "[https://wallpaperaccess.com/full/47178.jpg](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExOXh4cWozcGpkYzNhemx1MjA0cW40ZGwzanMwbmRhODVzN3g5OGg2NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/11kn6DFp9BNqWA/giphy.gif)" width = "800" height = "300" />
</p>

## Overview
This project was done as a B.Sc. (Honors) Physics final year dissertation. In this project, we aim to achieve a Variational Autoencoder latent space generalization across stellar surveys (SDSS and LAMOST) through a Multitask Learning Framework. The project investigates whether a latent space learned from one stellar spectroscopic survey encodes physically meaningful stellar features that generalise across the domain shift introduced by a different telescope, instrument, and calibration pipeline.

The generalization of latent representation is evaluated using a Multitask Learning Architecture where the encoder is connected to the downstream task heads: Regression head -> To estimate stellar atmospheric parameters, Classification head -> To classify stars into one of the seven MK (Morgan-Keenan) spectral types, Decoder -> To reconstruct the stellar spectra. This Framework is trained and validated on SDSS (Sloan Digital Sky Survey) stellar data and transfered onto LAMOST (Large Sky Area Multi-Object Fiber Spectroscopic Telescope) stellar data without any retraining to test the generalizability of latent representation.

To know in depth about this project, checkout the [Wiki](https://github.com/Alcatraz312/Survey-invariant-generalization/wiki). 

## Getting Started 
### Prerequisites
- Python 3.10
- CUDA-compatible GPU recommended (tested on NVIDIA RTX 3050 Ti, 12GB)
- Ubuntu 20.04+ or WSL2 on Windows
- Minimum 16GB RAM for loading both datasets simultaneously
- Comet ML account and API key for experiment tracking (set as environment 
  variable `comet_ml_key`)

### Data 
The spectral data files are not tracked in this repository due to size.

**SDSS DR18** — query via SQL at SDSS sky server https://skyserver.sdss.org/dr19/SearchTools/sql  
Use the queries described in Chapter 3 of the dissertation. Download FITS files 
using wget and parse into HDF5 using `scripts/preprocess_sdss.py`.

**LAMOST DR11 v2.0** — download from the LAMOST data portal at http://www.lamost.org/dr11/  
Query the LRS Stellar Parameter Catalogue for AFGK stars, download the bulk tar 
archive, and parse into HDF5 using `scripts/preprocess_lamost.py`.

Place the resulting HDF5 files at:
- `data/sdss_final_ready.h5`
- `data/lamost_combined.h5`

### Environment Variables 
Before running any training, set your Comet ML API key:

```bash
export comet_ml_key=your_api_key_here
```

If you do not want to use Comet ML, remove the `exp = comet_ml.start(...)` 
calls from `experiments/model_train.py`.

### Installation
1. Fork and clone the github repository to your local system:
```bash
git clone https://github.com/Alcatraz312/Survey-invariant-generalization.git
```
2. Go inside the project directory:
```bash
cd Survey-invariant-generalization
```
3. Create a virtual environment 
```bash
python -m venv <name of your environment>
```
4. Activate the virtual environment
* Linux:

```python
. <path_to_env>/bin/activate
```
* WSL:
```python
source <path_to_env>/bin/activate
```

* Windows (Powershell):

```python
<path_to_env>/Scripts/Activate.ps1
```

* Windows (Command Prompt):

```python
<path_to_env>/Scripts/Activate.bat
```

5. Upgrade pip to the latent version:
```bash
python.exe -m pip install --Upgrade pip
```
6. Install the required python package using pip:
```bash
pip install -r requirements.txt
```

## Contribution 
1. Fork the repository and create a new branch for your feature or bug fix.

2. Make your changes, and ensure that your code follows the PEP 8 style guide.

3. Write tests to cover your code if applicable.

4. Create a pull request with a clear description of your changes and why they are needed.

5. Your pull request will be reviewed, and once approved, it will be merged into the main branch.

## License
This project is licensed under the MIT License - see the [License](https://github.com/Alcatraz312/Survey-invariant-generalization/blob/main/LICENSE)

