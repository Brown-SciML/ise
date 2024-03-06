import os
import pandas as pd
import numpy as np

def get_all_filepaths(path: str, filetype: str = None, contains: str = None, not_contains: str = None):
    """Retrieves all filepaths for files within a directory. Supports subsetting based on filetype
    and substring search.

    Args:
        path (str): Path to directory to be searched.
        filetype (str, optional): File type to be returned (e.g. csv, nc). Defaults to None.
        contains (str, optional): Substring that files found must contain. Defaults to None.
        not_contains(str, optional): Substring that files found must NOT contain. Defaults to None.

    Returns:
        List[str]: list of files within the directory matching the input criteria.
    """
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        all_files += [os.path.join(dirpath, file) for file in filenames]

    if filetype:
        if filetype.lower() != "all":
            all_files = [file for file in all_files if file.endswith(filetype)]

    if contains:
        all_files = [file for file in all_files if contains in file]
        
    if not_contains:
        all_files = [file for file in all_files if not_contains not in file]
    
    return all_files

from netCDF4 import Dataset

def add_variable_to_nc(source_file_path, target_file_path, variable_name):
    """
    Copies a variable from a source NetCDF file to a target NetCDF file.

    Parameters:
    - source_file_path: Path to the source NetCDF file.
    - target_file_path: Path to the target NetCDF file.
    - variable_name: Name of the variable to be copied.

    Both files are assumed to have matching dimensions for the variable.
    """
    # Open the source NetCDF file in read mode
    with Dataset(source_file_path, 'r') as src_nc:
        # Check if the variable exists in the source file
        if variable_name in src_nc.variables:
            # Read the variable data and attributes
            variable_data = src_nc.variables[variable_name][:]
            variable_attributes = src_nc.variables[variable_name].ncattrs()
            
            # Open the target NetCDF file in append mode
            with Dataset(target_file_path, 'a') as target_nc:
                # Create or overwrite the variable in the target file
                if variable_name in target_nc.variables:
                    print(f"The '{variable_name}' variable already exists in the target file. Overwriting data.")
                    target_nc.variables[variable_name][:] = variable_data
                else:
                    # Create the variable with the same datatype and dimensions
                    variable = target_nc.createVariable(variable_name, src_nc.variables[variable_name].datatype, src_nc.variables[variable_name].dimensions)
                    
                    # Copy the variable attributes
                    for attr_name in variable_attributes:
                        variable.setncattr(attr_name, src_nc.variables[variable_name].getncattr(attr_name))
                    
                    # Assign the data to the new variable
                    variable[:] = variable_data
        else:
            print(f"'{variable_name}' variable not found in the source file.")

