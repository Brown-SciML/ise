from data.classes.AtmosphereForcing import AtmosphereForcing
from data.classes.GridSectors import GridSectors
from utils import get_all_filepaths
import pandas as pd
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

def aggregate_all(af_directory):
    """Loops through every NC file in the provided atmospheric forcing directory
    and 

    Args:
        af_directory (str): Directory containing atmospheric forcing files
    """
    start_time = time.time()
    filepaths = get_all_filepaths(path=af_directory, filetype='nc')
    filepaths = [f for f in filepaths if "1995-2100" in f]

    all_data = pd.DataFrame()
    for i, fp in enumerate(filepaths):
        print('')
        print(f'File {i+1} / {len(filepaths)}')
        print(f'File: {fp.split("/")[-1]}')
        print(f'Time since start: {(time.time()-start_time) // 60} minutes')
        atmosphere = aggregate_by_sector(fp)
        atmosphere.data = atmosphere.data[['pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly', 'smb_anomaly', 'ts_anomaly', 'regions', 'model',]]
        atmosphere.data.to_csv(f"{fp[:-3]}_sectoryeargrouped.csv")
        all_data = pd.concat([all_data, atmosphere.data])
        print(' -- ')
    
    all_data.to_csv(af_directory+'all_data.csv')