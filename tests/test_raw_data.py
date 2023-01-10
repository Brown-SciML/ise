import os
from paths import FORCING_DIRECTORY, GRIDS_DIRECTORY, ISMIP6_OUTPUT_DIRECTORY

# Test Forcing Data
def test_forcing_data_exists():
    assert os.path.exists(FORCING_DIRECTORY), "Forcing Directory doesn't exist. Download it from Globus Collection 'GHub-ISMIP6-Forcing'"
    
def test_forcing_AIS_directory():
    assert FORCING_DIRECTORY.endswith(r"AIS/"), "Directory must be specific to the AIS."
    
def test_forcing_correct_subfolders():
    assert all([sub in os.listdir(FORCING_DIRECTORY) for sub in ['Ocean_Forcing', 'Ice_Shelf_Fracture', 'Atmosphere_Forcing']]), "Forcing Directory does not contain the correct subdirectories."

def test_forcing_file_counts():
    assert len(os.listdir(f"{FORCING_DIRECTORY}/Ocean_Forcing")) == 26
    assert len(os.listdir(f"{FORCING_DIRECTORY}/Ice_Shelf_Fracture")) == 14
    assert len(os.listdir(f"{FORCING_DIRECTORY}/Atmosphere_Forcing")) == 19
    
    count = 0
    for _, _, files in os.walk(FORCING_DIRECTORY):
        count += len(files)
    
    assert count == 3673, "All files as found in Globus are not present. Functionality may be limited."



# Test Grids Data
def test_grids_data_exists():
    assert os.path.exists(GRIDS_DIRECTORY), "Grid Data Directory doesn't exist. Contact Helene Seroussi to get access."

    
def test_grids_AIS_directory():
    assert r'AIS' in GRIDS_DIRECTORY, "Directory must be specific to the AIS."
    
def test_grids_correct_files():
    assert all([sub in os.listdir(GRIDS_DIRECTORY) for sub in ['sectors_32km.nc', 'sectors_16km.nc', 'sectors_8km.nc', 'sectors_4km.nc']])
    
def test_grids_file_counts():
    assert len(os.listdir(GRIDS_DIRECTORY)) == 4
    assert all(substring.endswith(".nc") for substring in os.listdir(GRIDS_DIRECTORY))
    
    
    
    
# Test Output Data
def test_output_data_exists():
    assert os.path.exists(ISMIP6_OUTPUT_DIRECTORY), "ISMIP6 Output Directory doesn't exist. Download it from https://zenodo.org/record/3940766#.Y7yKwXZKhrp"
    
def test_output_AIS_directory():
    assert 'AIS' in ISMIP6_OUTPUT_DIRECTORY, "Directory must be specific to the AIS."
    
def test_output_correct_subfolders():
    assert all([sub in os.listdir(ISMIP6_OUTPUT_DIRECTORY) for sub in ['ComputedScalarsPaper',]])
    assert all([sub in os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper/") for sub in ['JPL1', 'VUB', 'NCAR', 'AWI', 'PIK', 'UTAS', 'UCIJPL', 'VUW', 'ULB', 'LSCE', 'DOE', 'ILTS_PIK', 'IMAU']])

def test_output_file_counts():
    assert len(os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper")) == 13
    assert len(os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper/JPL1")) == 1
    assert len(os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper/UCIJPL/ISSM")) == 21
    assert len(os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper/DOE/MALI")) == 10
    
    count = 0
    for _, _, files in os.walk(ISMIP6_OUTPUT_DIRECTORY):
        count += len(files)
    
    assert count == 3335, "All files as found in Zenodo are not present. Functionality may be limited."