"""Processing functions for ISMIP6 control experiments. Includes creating dataset for ctrl_proj values."""

from ise.utils.utils import get_all_filepaths
import xarray as xr
import pandas as pd
from tqdm import tqdm


def create_control_dataset(zenodo_directory: str, export_directory: str):
    """Creates dataset for lookup of control values. Can be added to prediction to adjust for 
    control group subtraction.

    Args:
        zenodo_directory (str): Directory containing Zenodo output files.
        export_directory (str): Directory to export processed outputs.
    """

    # traverse file structure to find directories with ctrl
    data_directory = f"{zenodo_directory}/ComputedScalarsPaper/"
    all_files = get_all_filepaths(data_directory, filetype="nc", contains="computed_ivaf")

    # for every folder in ctrl, get the computed_ivaf_ file
    ctrl_files = [f for f in all_files if '/ctrl_proj_' in f]

    all_ctrls = []
    for file in tqdm(ctrl_files):
        path_split = file.split('/')
        filename = path_split[-1]
        forcing_type = 'Open' if 'open' in filename else 'Standard'
        modelname = f"{path_split[-4]}_{path_split[-3]}"
        
        data = xr.open_dataset(file, decode_times=False)
        data = data.to_dataframe().reset_index().drop(columns=['ivaf', 'rhoi', 'rhow', 'ivaf_region_1', 'ivaf_region_2', 'ivaf_region_3'])
        data = pd.melt(data, id_vars="time")
        data['variable'] = data.variable.apply(lambda x: x.split('_')[-1])
        data['value'] = data.value / 1e9 / 362.5 # to sle
        data.columns = ['year', 'sectors', 'ctrl_sle']
        data['year'] = data.year.astype(int)
        data['modelname'] = modelname
        data["ocean_forcing"] = forcing_type
        all_ctrls.append(data)
        
    all_ctrls = pd.concat(all_ctrls)
    all_ctrls.to_csv(f'{export_directory}/ctrl_sle.csv', index=False)
    
    return all_ctrls
