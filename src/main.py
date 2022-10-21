from data.processing.aggregate_by_sector import aggregate_atmosphere, aggregate_ocean
from utils import get_all_filepaths
from utils import get_configs
cfg = get_configs()

data_directory = cfg['data']['path']
generate_atmospheric_forcing = True
generate_oceanic_forcing = True

if generate_atmospheric_forcing:
    af_directory = f"{data_directory}/Atmosphere_Forcing/"
    export_fp = af_directory + 'all_data_cols.csv'
    aggregate_atmosphere(af_directory, export=export_fp, model_in_columns=True, )
    
if generate_oceanic_forcing:
    of_directory = f"{data_directory}/Ocean_Forcing/"
    aggregate_ocean(of_directory, export=of_directory, model_in_columns=True, )
    
    
    
all_files = get_all_filepaths(path=f"{data_directory}/Ocean_Forcing/", filetype='nc')
files = [f for f in all_files if '1995-2100' in f]

stop = ''