import xarray as xr
from utils import check_input, get_configs
import os
import pandas as pd
import numpy as np
cfg = get_configs()

class GridSectors:
    def __init__(self, grid_size=8, filetype='nc'):
        check_input(grid_size, [4, 8, 16, 32])
        check_input(filetype.lower(), ['nc', 'csv'])
        self.grids_dir = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/ISMIP6_sectors/"
        
        if filetype.lower() == 'nc':
            self.path = self.grids_dir + f"sectors_{grid_size}km.nc"
            self.data = xr.open_dataset(self.path, decode_times=False)
            self = self._format_index()
        elif filetype.lower() == 'csv':
            self.path = self.grids_dir + f"sector_{grid_size}.csv"
            self.data = pd.read_csv(self.path)
        else:
            raise NotImplementedError('Only \"NetCDF\" and \"CSV\" are currently supported')
    
    def _netcdf_to_csv(self):
        if self.filetype != "NetCDF":
            raise AttributeError(f'Data type must be \"NetCDF\", received {self.datatype}.')
            
        csv_path = f"{self.path[:-3]}.csv"
        df = self.data.to_dataframe()
        df.to_csv(csv_path)

    def _to_dataframe(self):
        if not isinstance(self, pd.DataFrame):
            self.data = self.data.to_dataframe()
        return self

    def _format_index(self):
        self = self._to_dataframe()
        index_array = list(np.arange(0,761))
        self.data.index = pd.MultiIndex.from_product([index_array, index_array], names=['x', 'y'])
        return self