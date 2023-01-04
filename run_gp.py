from ise.models.gp import kernels
from ise.pipelines import training
import numpy as np
import pandas as pd
import time

processed_data = r"/users/pvankatw/emulator/untracked_folder/ml_data_directory"
gp_save_dir = r'/users/pvankatw/emulator/untracked_folder/gp/'
kernel = kernels.PowerExponentialKernel(exponential=0.1, ) + kernels.NuggetKernel()

start = time.time()
preds, std_prediction, metrics = training.train_gaussian_process(data_directory=processed_data, n=1000, 
                                features=['temperature',], sampling_method='first_n', 
                                kernel=kernel, verbose=True, save_directory=gp_save_dir)
finish = time.time()
print(f"Model 1 Total Time: {finish - start} seconds")



preds, std_prediction, metrics = training.train_gaussian_process(data_directory=processed_data, n=1000, 
                                features=['temperature', 'salinity'], sampling_method='first_n', 
                                kernel=kernel, verbose=True, save_directory=gp_save_dir)
finish = time.time()
print(f"Model 2 Total Time: {finish - start} seconds")



preds, std_prediction, metrics = training.train_gaussian_process(data_directory=processed_data, n=1000, 
                                features=['temperature', 'salinity', 'ice_shelf_fracture'], sampling_method='first_n', 
                                kernel=kernel, verbose=True, save_directory=gp_save_dir)
finish = time.time()
print(f"Model 3 Total Time: {finish - start} seconds")


start = time.time()
preds, std_prediction, metrics = training.train_gaussian_process(data_directory=processed_data, n=10000, 
                                features=['temperature',], sampling_method='first_n', 
                                kernel=kernel, verbose=True, save_directory=gp_save_dir)
finish = time.time()
print(f"Model 4 Total Time: {finish - start} seconds")


start = time.time()
preds, std_prediction, metrics = training.train_gaussian_process(data_directory=processed_data, n=10000, 
                                features=['temperature', 'salinity'], sampling_method='first_n', 
                                kernel=kernel, verbose=True, save_directory=gp_save_dir)
finish = time.time()
print(f"Model 5 Total Time: {finish - start} seconds")


start = time.time()
preds, std_prediction, metrics = training.train_gaussian_process(data_directory=processed_data, n=10000, 
                                features=['temperature', 'salinity', 'ice_shelf_fracture'], sampling_method='first_n', 
                                kernel=kernel, verbose=True, save_directory=gp_save_dir)
finish = time.time()
print(f"Model 6 Total Time: {finish - start} seconds")