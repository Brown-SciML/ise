import os
from paths import GRIDS_DIRECTORY

# Test Grids Data

def test_data_directory_exists():
    assert os.path.exists(GRIDS_DIRECTORY), "Grid Data Directory doesn't exist. Contact Helene Seroussi to get access."

def test_data_directory_exists():
    assert os.path.exists(GRIDS_DIRECTORY)
    
def test_AIS_directory():
    assert r'AIS' in GRIDS_DIRECTORY, "Directory must be specific to the AIS."
    
def test_correct_files():
    assert all([sub in os.listdir(GRIDS_DIRECTORY) for sub in ['sectors_32km.nc', 'sectors_16km.nc', 'sectors_8km.nc', 'sectors_4km.nc']])
    
def test_file_counts():
    assert len(os.listdir(GRIDS_DIRECTORY)) == 4
    assert all(substring.endswith(".nc") for substring in os.listdir(GRIDS_DIRECTORY))