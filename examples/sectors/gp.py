import sys
sys.path.append("../..")
from ise.sectors.models.gp import kernels
from ise.sectors.pipelines import training

GP_SAVE_DIR = r'/users/pvankatw/emulator/untracked_folder/gp/year_temp_salinity_n10000/'
PROCESSED_DATA = r"/users/pvankatw/emulator/untracked_folder/ml_data_directory"
kernel = kernels.PowerExponentialKernel() + kernels.NuggetKernel()
training.train_gaussian_process(data_directory=PROCESSED_DATA, n=10000, 
                                features=['year', 'temperature', 'salinity'], sampling_method='first_n', 
                                kernel=kernel, verbose=True, save_directory=GP_SAVE_DIR)