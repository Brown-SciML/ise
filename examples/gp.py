import sys
sys.path.append("..")
from ise.models.gp import kernels
from ise.pipelines import training
import numpy as np
import pandas as pd

processed_data = r"/users/pvankatw/emulator/untracked_folder/ml_data_directory"
gp_save_dir = r'/users/pvankatw/emulator/untracked_folder/gp/year_temp_salinity_n10000/'
kernel = kernels.PowerExponentialKernel() + kernels.NuggetKernel()
training.train_gaussian_process(data_directory=processed_data, n=10000, 
                                features=['year', 'temperature', 'salinity'], sampling_method='first_n', 
                                kernel=kernel, verbose=True, save_directory=gp_save_dir)