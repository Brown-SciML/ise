from data.processing.aggregate_by_sector import aggregate_atmosphere, aggregate_icecollapse, aggregate_ocean
from utils import get_configs
cfg = get_configs()

data_directory = cfg['data']['path']
export_dir = cfg['data']['export']

# TODO: Set these as config variables
generate_atmospheric_forcing = True
generate_oceanic_forcing = False
generate_icecollapse_forcing = True


if generate_atmospheric_forcing:
    af_directory = f"{data_directory}/Atmosphere_Forcing/"
    # TODO: refactor model_in_columns as aogcm_as_features
    aggregate_atmosphere(af_directory, export=export_dir, model_in_columns=True, )
    
if generate_oceanic_forcing:
    of_directory = f"{data_directory}/Ocean_Forcing/"
    aggregate_ocean(of_directory, export=export_dir, model_in_columns=True, )
    
if generate_icecollapse_forcing:
    ice_directory = f"{data_directory}/Ice_Shelf_Fracture"
    aggregate_icecollapse(ice_directory, export=export_dir, model_in_columns=True, )
    
# TODO: Add Ice Collapse Forcing Object & Aggregation
# TODO: Add Output Object
# TODO: Concatenate inputs to output object
    

stop = ''