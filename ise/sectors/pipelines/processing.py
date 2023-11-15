"""Pipeline function for processing forcing data, Zenodo output data, and merging them together for
use in modelling.
"""
import pandas as pd
from ise.sectors.data.processors.control import create_control_dataset
from ise.sectors.data.processors.forcings import process_forcings
from ise.sectors.data.processors.ismip6 import process_ismip6_outputs
from ise.sectors.data.processors.merge import merge_datasets


def process_data(
    forcing_directory: str,
    grids_directory: str,
    ismip6_output_directory: str,
    export_directory: str,
) -> pd.DataFrame:
    """Function for processing forcing data, Zenodo output data, and merging them together for
    use in modelling.

    Args:
        forcing_directory (str): Directory containing forcing files.
        grids_directory (str): Directory containing forcing files.
        ismip6_output_directory (str): Directory containing forcing
            files.
        export_directory (str): Directory containing forcing files.

    Returns:
        pd.DataFrame: master, Master dataset containing all processing
        outputs.
    """

    process_forcings(
        forcing_directory,
        grids_directory,
        export_directory,
        to_process="all",
        verbose=False,
    )
    process_ismip6_outputs(ismip6_output_directory, export_directory)
    master, inputs, outputs = merge_datasets(
        export_directory, export_directory, export_directory, include_icecollapse=False
    )
    create_control_dataset(ismip6_output_directory, export_directory)
    return master
