import xarray as xr
import numpy as np
import torch

def sum_by_sector(array, grid_file):
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    if isinstance(grid_file, str):
        grids = xr.open_dataset(grid_file)
        sector_name = 'sectors' if 'ais' in grid_file.lower() else 'ID'
    elif isinstance(grid_file, xr.Dataset):
        sector_name = 'ID' if 'Rignot' in grids.Description else 'sectors'
    else:
        raise ValueError("grid_file must be a string or an xarray Dataset.")
    
    if len(array.shape) == 3:
        num_timesteps = array.shape[0]
    elif len(array.shape) == 2:
        num_timesteps = 1
        array = array.reshape((1, array.shape[0], array.shape[1]))
    
    # if len(array.shape) == 3:
    #     grids = grids.expand_dims(dim={'time': num_timesteps})
    sectors = grids[sector_name].values
    
    ice_sheet = 'AIS' if 761 in array.shape else 'GIS'
    num_sectors = 18 if ice_sheet == 'AIS' else 6
    
    sums_by_sector = np.zeros((num_timesteps, num_sectors))
    for i in range(array.shape[0]):
        for sector in range(1, num_sectors+1):
            sums_by_sector[i, sector-1] = np.sum(array[i, :, :][sectors == sector])
    return sums_by_sector
        

def mean_squared_error_sector(sum_sectors_true, sum_sectors_pred):
    return np.mean((sum_sectors_true - sum_sectors_pred)**2)