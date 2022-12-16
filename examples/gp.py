import sys
sys.path.append("..")
from ise.models.gp import kernels
from ise.pipelines import training
import numpy as np
import pandas as pd

processed_data = r"/users/pvankatw/emulator/untracked_folder/ml_data_directory"
gp_save_dir = r'/users/pvankatw/emulator/untracked_folder/gp/'
kernel = kernels.PowerExponentialKernel(exponential=0.1, ) + kernels.NuggetKernel()
training.train_gaussian_process(data_directory=processed_data, n=1000, 
                                features=['temperature', 'salinity'], sampling_method='first_n', 
                                kernel=kernel, verbose=True, save_directory=gp_save_dir)


preds = np.array(pd.read_csv(f'{gp_save_dir}/preds.csv')).squeeze()
std = np.array(pd.read_csv(f'{gp_save_dir}/std.csv')).squeeze()
test_labels = pd.read_csv(f'{processed_data}/ts_test_labels.csv')
test_labels.to_csv(f"{gp_save_dir}/test_labels.csv")