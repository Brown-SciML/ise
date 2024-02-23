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

