import os
from paths import ISMIP6_OUTPUT_DIRECTORY


def test_data_directory_exists():
    assert os.path.exists(ISMIP6_OUTPUT_DIRECTORY), "ISMIP6 Output Directory doesn't exist. Download it from https://zenodo.org/record/3940766#.Y7yKwXZKhrp"
    
def test_AIS_directory():
    assert 'AIS' in ISMIP6_OUTPUT_DIRECTORY, "Directory must be specific to the AIS."
    
def test_correct_subfolders():
    assert all([sub in os.listdir(ISMIP6_OUTPUT_DIRECTORY) for sub in ['ComputedScalarsPaper',]])
    assert all([sub in os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper/") for sub in ['JPL1', 'VUB', 'NCAR', 'AWI', 'PIK', 'UTAS', 'UCIJPL', 'VUW', 'ULB', 'LSCE', 'DOE', 'ILTS_PIK', 'IMAU']])

def test_file_counts():
    assert len(os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper")) == 13
    assert len(os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper/JPL1")) == 1
    assert len(os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper/UCIJPL/ISSM")) == 21
    assert len(os.listdir(f"{ISMIP6_OUTPUT_DIRECTORY}/ComputedScalarsPaper/DOE/MALI")) == 10
    
    count = 0
    for _, _, files in os.walk(ISMIP6_OUTPUT_DIRECTORY):
        count += len(files)
    
    assert count == 3335, "All files as found in Globus are not present. Functionality may be limited."