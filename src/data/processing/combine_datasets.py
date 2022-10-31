import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from utils import get_all_filepaths, get_configs
import json
cfg = get_configs()


export_dir = cfg['data']['export']
with_ice = cfg['processing']['include_icecollapse']

with open(cfg['data']['ismip6_experiments_json']) as experiments:
    ismip6_experiments = json.load(experiments)



def combine_datasets(processed_data_dir=export_dir, include_icecollapse=with_ice, export=export_dir):
    
    try:
        af = pd.read_csv(f"{processed_data_dir}/atmospheric_forcing.csv")
        ice = pd.read_csv(f"{processed_data_dir}/ice_collapse.csv")
        salinity = pd.read_csv(f"{processed_data_dir}/salinity.csv")
        temp = pd.read_csv(f"{processed_data_dir}/temperature.csv")
        tf = pd.read_csv(f"{processed_data_dir}/thermal_forcing.csv")
        outputs = pd.read_csv(f"{processed_data_dir}/outputs.csv")
    except FileNotFoundError:
        raise FileNotFoundError('Files not found, make sure to run all processing functions.')
    
    ocean = salinity
    ocean['aogcm'] = ocean['aogcm'].apply(format_aogcms)
    af['aogcm'] = af['aogcm'].apply(format_aogcms)
    for data in [temp, tf,]:
        data['aogcm'] = data['aogcm'].apply(format_aogcms)
        ocean = pd.merge(ocean, data, on=['sectors', 'year', 'aogcm', 'regions'], how="outer")
    ocean = ocean.drop_duplicates()
    ocean = ocean[['sectors', 'regions', 'year', 'aogcm', 'salinity', 'temperature', 'thermal_forcing']]
    
    af['aogcm'] = af['aogcm'].apply(format_aogcms)
    ice['aogcm'] = ice['aogcm'].apply(format_aogcms)
    
    
    inputs = pd.merge(ocean, af, on=['sectors', 'year', 'aogcm', 'regions'], how="inner")
    
    if include_icecollapse:
        inputs = pd.merge(inputs, ice, on=['sectors', 'year', 'aogcm', 'regions'], how="inner")
        
    outputs['experiment'], outputs['aogcm'], outputs['scenario'], outputs['ocean_forcing'], outputs['ocean_sensitivity'], outputs['ice_shelf_fracture'], outputs['tier'] = zip(*outputs['exp_id'].map(exp_to_attributes))
    
    master = pd.merge(inputs, outputs, on=['year', 'sectors', 'aogcm'])
    
    if export:
        master.to_csv(f"{export}/master.csv")
        inputs.to_csv(f"{export}/inputs.csv")
        outputs.to_csv(f"{export}/outputs.csv")
    
    return master, inputs, outputs
    
    
    

def exp_to_attributes(x, ):
    try:
        attributes = ismip6_experiments[x]
        return attributes['Experiment'], attributes['AOGCM'], attributes['Scenario'], attributes['Ocean forcing'], attributes['Ocean sensitivity'], attributes['Ice shelf fracture'], attributes['Tier'] 
    except:
        pass


def format_aogcms(x):
    x = x.lower().replace(".", "").replace("-", "_")
    ssp = 'ssp' in x
    rcp = 'rcp' in x
    try:
        try:
            correct_format = re.search("(ssp|rcp)\d{2,3}", x).group()
        except AttributeError:
            # no matche
            numeric = re.search("\d{2,3}", x).group()
            x = x[:-2]
            if x.endswith('_'):
                x += f"rcp{numeric}"
            else:
                x += f"_rcp{numeric}"
                
    except AttributeError:
        # if none of the above worked, just skip it
        pass
    
    x = x.replace("_1", "")
    if x == "ukesm1_0_ll":
        x += "_ssp585"
    return x