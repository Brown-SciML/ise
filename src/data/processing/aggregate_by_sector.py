from data.classes.AtmosphereForcing import AtmosphereForcing
from data.classes.GridSectors import GridSectors
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
    grids = GridSectors(grid_size=8)

    # Load in Atmospheric forcing data and add the sector numbers to it
    af = AtmosphereForcing(path=path)
    af = af.add_sectors(grids)

    # Group the dataset and assign model column to the model simulation
    af.data = af.data.groupby(['sectors', 'year']).mean()
    af.data['model'] = af.model
    return af

def aggregate_all(af_directory, export, model_in_columns=False, ):
    """Loops through every NC file in the provided atmospheric forcing directory
    from 1995-2100 and applies the aggregate_by_sector function. It then outputs
    the concatenation of all processed data to all_data.csv 

    Args:
        af_directory (str): Directory containing atmospheric forcing files
    """
    start_time = time.time()

    # Get all NC files that contain data from 1995-2100
    filepaths = get_all_filepaths(path=af_directory, filetype='nc')
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
        atmosphere = aggregate_by_sector(fp)

        # Handle files that don't have mrro_anomaly input (ISPL RCP 85?)
        try:
            atmosphere.data['mrro_anomaly']
        except KeyError:
            atmosphere.data['mrro_anomaly'] = np.nan

        # Keep selected columns and output each file individually
        atmosphere.data = atmosphere.data[['pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly', 'smb_anomaly', 'ts_anomaly', 'regions', 'model',]]
        atmosphere.data.to_csv(f"{fp[:-3]}_sectoryeargrouped.csv")

        # meanwhile, create a concatenated dataset
        all_data = pd.concat([all_data, atmosphere.data])
        print(' -- ')
    
    
    if model_in_columns:
        separate_model_dataframes = [y for x, y in all_data.groupby('model')]
        
        # Change columns names in each dataframe
        for df in separate_model_dataframes:
            model = df.model.iloc[0]
            df.columns = [f"{x}_{model}" if x not in ['sectors', 'year', 'region', 'model'] else x for x in df.columns ]
            
        # Merge dataframes together on common columns [sectors, year], resulting in 
        # one dataframe with sector, year, region, and columns for each model variables
        all_data = model_dataframes[0]
        all_data = all_data.drop(columns=['model'])
        for df in model_dataframes[1:]:
            df = df.drop(columns=['model'])
            all_data = pd.merge(all_data, df, on=['sectors', 'year',])
        all_data = all_data[[c for c in all_data.columns if 'region' not in c]]
        all_data['region'] = model_dataframes[0]['regions_CESM2_ssp585'].reset_index(drop=True)
    
    if export:
        all_data.to_csv(export)