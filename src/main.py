from data.processing.aggregate_by_sector import aggregate_atmosphere, aggregate_icecollapse, aggregate_ocean
from data.processing.process_outputs import process_repository
from utils import get_configs, get_all_filepaths
cfg = get_configs()

data_directory = cfg['data']['forcing']
zenodo_directory = cfg['data']['output']
export_dir = cfg['data']['export']

# TODO: Set these as config variables
generate_atmospheric_forcing = False
generate_oceanic_forcing = False
generate_icecollapse_forcing = False
process_outputs = True


if generate_atmospheric_forcing:
    af_directory = f"{data_directory}/Atmosphere_Forcing/"
    # TODO: refactor model_in_columns as aogcm_as_features
    aggregate_atmosphere(af_directory, export=export_dir, model_in_columns=False, )
    
if generate_oceanic_forcing:
    of_directory = f"{data_directory}/Ocean_Forcing/"
    aggregate_ocean(of_directory, export=export_dir, model_in_columns=False, )
    
if generate_icecollapse_forcing:
    ice_directory = f"{data_directory}/Ice_Shelf_Fracture"
    aggregate_icecollapse(ice_directory, export=export_dir, model_in_columns=False, )
    
if process_outputs:
    process_repository(zenodo_directory, export_filepath=f"{export_dir}/outputs.csv")
    
# TODO: Add Ice Collapse Forcing Object & Aggregation
# TODO: Add Output Object
# TODO: Concatenate inputs to output object

# files = get_all_filepaths(data_directory, 'csv')
# fs = [f for f in files if 'sectoryeargrouped' in f]

# import pandas as pd
# all_data = pd.DataFrame()
# for f in fs:
#     temp = pd.read_csv(f)
#     temp = temp.groupby(['sectors', 'year', 'model']).mean()
#     print(len(temp))
#     all_data = pd.concat([all_data, temp],)




    

stop = ''