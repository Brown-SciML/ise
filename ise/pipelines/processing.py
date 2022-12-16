"""Pipeline function for processing forcing data, Zenodo output data, and merging them together for
use in modelling."""
import pandas as pd
from ise.data.processors.forcings import process_forcings
from ise.data.processors.ismip6 import process_ismip6_outputs
from ise.data.processors.merge import merge_datasets

def process_data(forcing_directory: str, grids_directory: str, 
                 ismip6_output_directory: str, export_directory: str) -> pd.DataFrame:
    """Function for processing forcing data, Zenodo output data, and merging them together for
    use in modelling.

    :param forcing_directory: Directory containing forcing files.
    :type forcing_directory: str
    :param grids_directory: Directory containing forcing files.
    :type grids_directory: str
    :param ismip6_output_directory: Directory containing forcing files.
    :type ismip6_output_directory: str
    :param export_directory: Directory containing forcing files.
    :type export_directory: str
    :return: master, Master dataset containing all processing outputs.
    :rtype: pd.DataFrame
    """

    process_forcings(forcing_directory, grids_directory, export_directory, 
                     to_process='all', verbose=False)
    process_ismip6_outputs(ismip6_output_directory, export_directory)
    master, inputs, outputs = merge_datasets(export_directory, export_directory, 
                                             export_directory, include_icecollapse=False)
    return master
    