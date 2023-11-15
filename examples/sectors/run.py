import sys
sys.path.append("../..")

from ise.sectors.pipelines import processing, feature_engineering, training
from ise.sectors.models.gp import kernels

FORCING_DIRECTORY = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/"
ISMIP6_OUTPUT_DIRECTORY = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Zenodo_Outputs/"
GRIDS_DIRECTORY = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/ISMIP6_sectors/"
PROCESSED_FORCING_OUTPUTS = r"/users/pvankatw/emulator/untracked_folder/processed_forcing_outputs/"
ML_DATA_DIRECTORY = r"/users/pvankatw/emulator/untracked_folder/ml_data_directory/"
SAVED_MODEL_PATH = r"/users/pvankatw/emulator/untracked_folder/saved_models/"

print('1/4: Processing Data')
master = processing.process_data(
    forcing_directory=FORCING_DIRECTORY, 
    grids_directory=GRIDS_DIRECTORY,
    ismip6_output_directory=ISMIP6_OUTPUT_DIRECTORY,
    export_directory=PROCESSED_FORCING_OUTPUTS,
)

print('2/4: Feature Engineering')
feature_engineering.feature_engineer(
    data_directory=PROCESSED_FORCING_OUTPUTS,
    time_series=True,
    export_directory=ML_DATA_DIRECTORY,
)

print('3/4: Training Neural Network Model')
model, metrics, test_preds = training.train_timeseries_network(
    data_directory=ML_DATA_DIRECTORY,
    save_model=SAVED_MODEL_PATH,
    verbose=True,
    epochs=10,
)

print('4/4: Training Gaussian Process Model')
kernel = kernels.PowerExponentialKernel(exponential=1.9, ) + kernels.NuggetKernel()
preds, std_prediction, metrics = training.train_gaussian_process(
    data_directory=ML_DATA_DIRECTORY,
    n=1000,
    kernel=kernel,
    features=['temperature', 'salinity'],
    sampling_method='first_n',
)
