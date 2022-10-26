import xarray as xr
import numpy as np
import pandas as pd
from utils import get_all_filepaths, get_configs
import os
from itertools import compress
cfg = get_configs()

data_directory = cfg['data']['forcing']
output_directory = cfg['data']['output']
variables = ['iareafl', 'iareagr', 'icearea', 'ivol', 'ivaf', 'smb', 'snbgr', 'bmbfl']

def get_sector(x):
    x = x.split("_")
    if len(x) == 1 or 'region' in x:
        return np.nan
    return int(x[-1])

def process_repository(zenodo_directory, export_filepath=None):
    files = get_all_filepaths(path, filetype='nc', contains='minus_ctrl_proj')
    groups_dir = f"{output_directory}/ComputedScalarsPaper/"
    all_groups = os.listdir(groups_dir)
    all_data = pd.DataFrame()
    for group in all_groups:
        group_path = f"{groups_dir}/{group}/"
        for model in os.listdir(group_path):
            model_path = f"{group_path}/{model}/"
            for exp in [f for f in os.listdir(model_path) if f not in ('historical', 'ctr', 'ctr_proj', 'asmb', 'abmb')]:
                exp_path = f"{model_path}/{exp}/"
                processed_experiment = process_experiment(exp_path)
                all_data = pd.concat([all_data, processed_experiment])
                
    if export:
        all_data.to_csv(export_filepath)
        
            
        
def process_experiment(experiment_directory):
    files = get_all_filepaths(experiment_directory, filetype='nc', contains='minus_ctrl_proj')
    
    all_data = process_single_file(file[0])
    for file in files[1:]:
        temp = process_single_file(file)
        all_data = pd.merge(all_data, temp, on=['time', 'sector', 'groupname', 'modelname', 'exp_id', 'rhoi', 'rhow'], how='outer')
        
    return all_data
        

def process_single_file(path):
    var = list(compress(variables, [v in fp for v in variables]))
    data = xr.open_dataset(path, decode_times=False)
    
    fp_split = path.split('/')
    groupname = fp_split[-4]
    modelname = fp_split[-3]
    exp_id = fp_split[-2]
    
    rhoi = data.rhoi
    rhow = data.rhow
    data = data.drop(labels=['rhoi', 'rhow'])
    
    data = data.to_dataframe().reset_index()
    data['time'] = np.floor(data['time']).astype(int)
    data = pd.melt(data, id_vars='time')

    data['sector'] = data.variable.apply(get_sector)
    data = data.dropna().drop(columns=['variable'])
    data[var] = data['value']
    data = data.drop(columns=['value'])
    
    data['groupname'] = groupname
    data['modelname'] = modelname
    data['exp_id'] = exp_id
    data['rhoi'] = rhoi.values
    data['rhow'] = rhow.values
    
    return data






