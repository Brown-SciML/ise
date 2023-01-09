import os
from paths import FORCING_DIRECTORY

# Test Forcing Data

def test_data_directory_exists():
    assert os.path.exists(FORCING_DIRECTORY), "Forcing Directory doesn't exist. Download it from Globus Collection 'GHub-ISMIP6-Forcing'"
    
def test_data_directory_exists():
    assert os.path.exists(FORCING_DIRECTORY), "Forcing Directory does not exist."
    
def test_AIS_directory():
    assert FORCING_DIRECTORY.endswith(r"AIS/"), "Directory must be specific to the AIS."
    
def test_correct_subfolders():
    assert all([sub in os.listdir(FORCING_DIRECTORY) for sub in ['Ocean_Forcing', 'Ice_Shelf_Fracture', 'Atmosphere_Forcing']]), "Forcing Directory does not contain the correct subdirectories."

def test_file_counts():
    assert len(os.listdir(f"{FORCING_DIRECTORY}/Ocean_Forcing")) == 26
    assert len(os.listdir(f"{FORCING_DIRECTORY}/Ice_Shelf_Fracture")) == 14
    assert len(os.listdir(f"{FORCING_DIRECTORY}/Atmosphere_Forcing")) == 19
    
    count = 0
    for _, _, files in os.walk(FORCING_DIRECTORY):
        count += len(files)
    
    assert count == 3673, "All files as found in Globus are not present. Functionality may be limited."