import os
import pandas as pd
from paths import PROCESSED_FORCING_OUTPUTS

# Test Forcing Data
def test_processed_data_exists():
    assert os.path.exists(PROCESSED_FORCING_OUTPUTS), "Processed forcings directory doesn't exist. Run the processing pipelines found in ise.pipelines.processing."
    
def test_processed_correct_files():
    assert all([sub in os.listdir(PROCESSED_FORCING_OUTPUTS) for sub in ['thermal_forcing.csv', 'salinity.csv', 'temperature.csv', 'master.csv', 'ice_collapse.csv', 'atmospheric_forcing.csv']]), "Forcing Directory does not contain the correct subdirectories."

thermal_forcing = pd.read_csv(f"{PROCESSED_FORCING_OUTPUTS}/thermal_forcing.csv")
salinity = pd.read_csv(f"{PROCESSED_FORCING_OUTPUTS}/salinity.csv")
temperature = pd.read_csv(f"{PROCESSED_FORCING_OUTPUTS}/temperature.csv")
master = pd.read_csv(f"{PROCESSED_FORCING_OUTPUTS}/master.csv", low_memory=False)
ice_collapse = pd.read_csv(f"{PROCESSED_FORCING_OUTPUTS}/ice_collapse.csv")
atmospheric_forcing = pd.read_csv(f"{PROCESSED_FORCING_OUTPUTS}/atmospheric_forcing.csv")
processed_data = [thermal_forcing, salinity, temperature, master, ice_collapse, atmospheric_forcing]
    
def test_processed_nonempty():
    assert all([not dataset.empty for dataset in processed_data]), "One of the processed files is empty."

def test_processed_attributes():
    # Test each dataset for correct columns
    assert all([column in dataset for dataset in processed_data for column in ['year', 'aogcm', 'sectors']]), "Year, AOGCM, or Sectors columns are missing from processed data."
    assert 'temperature' in temperature.columns, "Temperature column missing from temperature dataset."
    assert 'salinity' in salinity.columns, "Salinity column missing from salinity dataset."
    assert all([column in master.columns for column in ['salinity', 'temperature', 'thermal_forcing', 'evspsbl_anomaly']]), "Master dataset does not contain all columns."