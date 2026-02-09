import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt 
from astropy.io import fits

folder_directory = "/home/arbiter/projects/data_downloads"

folder_list = os.listdir(folder_directory)
print(folder_list)