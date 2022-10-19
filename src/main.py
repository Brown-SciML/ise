from data.processing.aggregate_by_sector import aggregate_all
from utils import get_all_filepaths
import xarray as xr
import pandas as pd
from utils import get_configs
cfg = get_configs()

data_directory = cfg['data']['path']
generate_atmospheric_forcing = False

if generate_atmospheric_forcing:
    af_directory = f"{data_directory}/Atmosphere_Forcing/"
    export_dir = af_directory + 'all_data_cols.csv'
    aggregate_all(af_directory, export=export_dir, model_in_columns=True)
    
    
    
all_files = get_all_filepaths(path=f"{data_directory}/Ocean_Forcing/", filetype='nc')
files = [f for f in all_files if '1995-2100' in f]

stop = ''