import xarray as xr
import numpy as np
np.random.seed(10)
import pandas as pd
from ise.utils.utils import get_all_filepaths, get_configs
import os
from itertools import compress
from tqdm import tqdm

variables = ['iareafl', 'iareagr', 'icearea', 'ivol', 'ivaf', 'smb', 'smbgr', 'bmbfl']

def process_ismip6_outputs(zenodo_directory, export_directory):
    process_repository(zenodo_directory, export_filepath=f"{export_directory}/ismip6_outputs.csv")

def get_sector(x):
    x = x.split("_")
    if len(x) == 1 or 'region' in x:
        return np.nan
    return int(x[-1])

def process_repository(zenodo_directory, export_filepath=None):
    # files = get_all_filepaths(zenodo_directory, filetype='nc', contains='minus_ctrl_proj')
    groups_dir = f"{zenodo_directory}/ComputedScalarsPaper/"
    all_groups = os.listdir(groups_dir)
    all_data = pd.DataFrame()
    
    # For each modeling group
    for group in tqdm(all_groups, total=len(all_groups)):
        group_path = f"{groups_dir}/{group}/"
        
        # For each model they submitted
        for model in os.listdir(group_path):
            model_path = f"{group_path}/{model}/"
            
            # For each experiment they carried out...
            not_experiments = ('historical', 'ctr', 'ctr_proj', 'asmb', 'abmb', 'ctrl_proj_std', 'hist_std', 'hist_open', 'ctrl_proj_open')
            all_experiments = [f for f in os.listdir(model_path) if f not in not_experiments]
            for exp in all_experiments:
                exp_path = f"{model_path}/{exp}/"
                processed_experiment = process_experiment(exp_path)
                all_data = pd.concat([all_data, processed_experiment])
                
    if export_filepath:
        all_data.to_csv(export_filepath, index=False)
        
    return all_data
        
            
        
def process_experiment(experiment_directory):
    files = get_all_filepaths(experiment_directory, filetype='nc', contains='minus_ctrl_proj')
    
    all_data = process_single_file(files[0])
    for file in files[1:]:
        temp = process_single_file(file)
        all_data = pd.merge(all_data, temp, on=['year', 'sectors', 'groupname', 'modelname', 'exp_id', 'rhoi', 'rhow'], how='outer')
        
    return all_data
        

def process_single_file(path):
    
    var = list(compress(variables, [v in path for v in variables]))
    
    # ! Fix this: getting confused with "smb" vs "smb" using "in" operator -- come up with better way
    if len(var) > 1:
        var = var[1]
    else:
        var = var[0]
        
    data = xr.open_dataset(path, decode_times=False)
    
    fp_split = [f for f in path.split('/') if f != '']
    groupname = fp_split[-4]
    modelname = fp_split[-3]
    exp_id = fp_split[-2]
    
    try:
        rhoi = data.rhoi.values
        rhow = data.rhow.values
        data = data.drop(labels=['rhoi', 'rhow'])
    except AttributeError: 
        rhoi = np.nan
        rhow = np.nan

    data = data.to_dataframe().reset_index()
    data['year'] = np.floor(data['time']).astype(int)
    data = data.drop(columns='time')
    data = pd.melt(data, id_vars='year')

    data['sectors'] = data.variable.apply(get_sector)
    data = data.dropna().drop(columns=['variable'])
    data[var] = data['value']
    data = data.drop(columns=['value'])
    
    data['groupname'] = groupname
    data['modelname'] = modelname
    data['exp_id'] = exp_id
    data['rhoi'] = rhoi
    data['rhow'] = rhow
    
    return data






