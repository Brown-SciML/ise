from data.classes.AtmosphereForcing import AtmosphereForcing
from data.classes.GridSectors import GridSectors
from data.classes.OceanForcing import OceanForcing
from utils import get_all_filepaths
import pandas as pd
import numpy as np
import time

def aggregate_by_sector(path):
    """Takes a atmospheric forcing dataset, adds sector numbers to it,
    and gets aggregate data based on sector and year. Returns atmospheric
    forcing data object.

    Args:
        path (str): path to atmospheric forcing nc file

    Returns:
        Obj: AtmosphereForcing instance with aggregated data
    """
    # Load grid data with 8km grid size
    
    print('')

    # Load in Atmospheric forcing data and add the sector numbers to it
    if 'Atmosphere' in path:
        grids = GridSectors(grid_size=8)
        forcing = AtmosphereForcing(path=path)
        
    elif 'Ocean' in path:
        grids = GridSectors(grid_size=8, format_index=False)
        forcing = OceanForcing(model_dir=path)

    forcing = forcing.add_sectors(grids)
    
    
    # SOMEHOW ONLY SECTOR THREE IS SHOWING UP. IT HAPPENS BEFORE HERE (I THINK IN ADD_SECTORS)
    
    # Group the dataset and assign model column to the model simulation
    if forcing.forcing_type == 'atmosphere':
        forcing.data = forcing.data.groupby(['sectors', 'year']).mean()
        forcing.data['model'] = forcing.model
    elif forcing.forcing_type == 'ocean':
        forcing.salinity_data = forcing.salinity_data.groupby(['sectors', 'year']).mean()
        forcing.salinity_data['model'] = forcing.model
        forcing.temperature_data = forcing.temperature_data.groupby(['sectors', 'year']).mean()
        forcing.temperature_data['model'] = forcing.model
        forcing.thermal_forcing_data = forcing.thermal_forcing_data.groupby(['sectors', 'year']).mean()
        forcing.thermal_forcing_data['model'] = forcing.model        
    
    return forcing

def aggregate_atmosphere(directory, export, model_in_columns=False,):
    """Loops through every NC file in the provided forcing directory
    from 1995-2100 and applies the aggregate_by_sector function. It then outputs
    the concatenation of all processed data to all_data.csv 

    Args:
        directory (str): Directory containing forcing files
    """
    start_time = time.time()

    # Get all NC files that contain data from 1995-2100
    filepaths = get_all_filepaths(path=directory, filetype='nc')
    filepaths = [f for f in filepaths if "1995-2100" in f]
    

    # Useful progress prints
    print(f"Files to be processed...")
    print([f.split("/")[-1] for f in filepaths])

    # Loop over each file specified above
    all_data = pd.DataFrame()
    for i, fp in enumerate(filepaths):
        print('')
        print(f'File {i+1} / {len(filepaths)}')
        print(f'File: {fp.split("/")[-1]}')
        print(f'Time since start: {(time.time()-start_time) // 60} minutes')

        # attach the sector to the data and groupby sectors & year
        forcing = aggregate_by_sector(fp)

        # Handle files that don't have mrro_anomaly input (ISPL RCP 85?)
        try:
            forcing.data['mrro_anomaly']
        except KeyError:
            forcing.data['mrro_anomaly'] = np.nan

        # Keep selected columns and output each file individually
        forcing.data = forcing.data[['pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly', 'smb_anomaly', 'ts_anomaly', 'regions', 'model',]]
    
        forcing.data.to_csv(f"{fp[:-3]}_sectoryeargrouped.csv")

        # meanwhile, create a concatenated dataset
        all_data = pd.concat([all_data, forcing.data])
            
        print(' -- ')
    
    
    
    # TODO Put this in its own function? Make it so that it recognizes how many times it needs to iterate
    if model_in_columns:
        separate_model_dataframes = [y for x, y in all_data.groupby('model')]
        
        # Change columns names in each dataframe
        for df in separate_model_dataframes:
            model = df.model.iloc[0]
            df.columns = [f"{x}_{model}" if x not in ['sectors', 'year', 'region', 'model'] else x for x in df.columns ]
            
        # Merge dataframes together on common columns [sectors, year], resulting in 
        # one dataframe with sector, year, region, and columns for each model variables
        all_data = separate_model_dataframes[0]
        all_data = all_data.drop(columns=['model'])
        for df in separate_model_dataframes[1:]:
            df = df.drop(columns=['model'])
            all_data = pd.merge(all_data, df, on=['sectors', 'year',])
        all_data = all_data[[c for c in all_data.columns if 'region' not in c]]
        all_data['region'] = separate_model_dataframes[0]['regions_CESM2_ssp585'].reset_index(drop=True)
        all_data = all_data.drop_duplicates()  # TODO: See why there are duplicates -- until then, this works
    
    if export:
        all_data.to_csv(export)
        
        
def aggregate_ocean(directory, export, model_in_columns=False, ):
    """Loops through every NC file in the provided forcing directory
    from 1995-2100 and applies the aggregate_by_sector function. It then outputs
    the concatenation of all processed data to all_data.csv 

    Args:
        directory (str): Directory containing forcing files
    """
    start_time = time.time()

    # Get all NC files that contain data from 1995-2100
    filepaths = get_all_filepaths(path=directory, filetype='nc')
    filepaths = [f for f in filepaths if "1995-2100" in f]
    
    # In the case of ocean forcings, use the filepaths of the files to determine
    # which directories need to be used for OceanForcing processing. Change to
    # those directories rather than individual files.
    models = list(set([f.split('/')[-3] for f in filepaths]))
    filepaths = [f"{directory}/{model}/" for model in models]

    # Useful progress prints
    print(f"Files to be processed...")
    print([f.split("/")[-2] for f in filepaths])

    # Loop over each directory specified above
    salinity_data = pd.DataFrame()
    temperature_data = pd.DataFrame()
    thermal_forcing_data = pd.DataFrame()
    for i, fp in enumerate(filepaths):
        print('')
        print(f'Directory {i+1} / {len(filepaths)}')
        print(f'Directory: {fp.split("/")[-2]}')
        print(f'Time since start: {(time.time()-start_time) // 60} minutes')

        # attach the sector to the data and groupby sectors & year
        forcing = aggregate_by_sector(fp)

        forcing.salinity_data = forcing.salinity_data[['salinity', 'regions', 'model']]
        forcing.temperature_data = forcing.temperature_data[['temperature', 'regions', 'model']]
        forcing.thermal_forcing_data = forcing.thermal_forcing_data[['thermal_forcing', 'regions', 'model']]
        
        print(forcing.salinity_data)
        print(forcing.temperature_data)
        print(forcing.thermal_forcing_data)
        
        
        # meanwhile, create a concatenated dataset
        salinity_data = pd.concat([salinity_data, forcing.salinity_data])
        temperature_data = pd.concat([temperature_data, forcing.temperature_data])
        thermal_forcing_data = pd.concat([thermal_forcing_data, forcing.thermal_forcing_data])
        
        salinity_data.to_csv(export+'/_salinity.csv')
        temperature_data.to_csv(export+'/_temperature.csv')
        thermal_forcing_data.to_csv(export+'/_thermal_forcing.csv')
        
    print(' -- ')
    
    print('Creating column variables for individual models...')
    
    # ! BUG: somowhow the last three rows (2098-2100) get dropped off. Not sure exactly where but i think
    # ! its here because it exists in _salinity but not after this chunk.
    if model_in_columns:
        # For each concatenated dataset
        datasets = [salinity_data, temperature_data, thermal_forcing_data]
        labels = ['salinity', 'temperature', 'thermal_forcing']
        for i, all_data in enumerate(datasets):
            separate_model_dataframes = [y for x, y in all_data.groupby('model')]
            
            # Change columns names in each dataframe
            for df in separate_model_dataframes:
                model = df.model.iloc[0]
                df.columns = [f"{x}_{model}" if x not in ['sectors', 'year', 'region', 'model'] else x for x in df.columns ]
                
            # Merge dataframes together on common columns [sectors, year], resulting in 
            # one dataframe with sector, year, region, and columns for each model variables
            all_data = separate_model_dataframes[0]
            all_data = all_data.drop(columns=['model'])
            for df in separate_model_dataframes[1:]:
                df = df.drop(columns=['model'])
                all_data = pd.merge(all_data, df, on=['sectors', 'year',])
            region_cols = [c for c in all_data.columns if 'region' in c]
            all_data = all_data[[c for c in all_data.columns if 'region' not in c]]
            all_data['region'] = separate_model_dataframes[0][region_cols[0]].reset_index(drop=True)
            all_data = all_data.drop_duplicates()
            
            if export:
                all_data.to_csv(f"{export}/{labels[i]}.csv")
    
    else:
        if export:
            salinity_data.to_csv(export+'/salinity.csv')
            temperature_data.to_csv(export+'/temperature.csv')
            thermal_forcing_data.to_csv(export+'/thermal_forcing.csv')