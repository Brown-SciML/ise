import xarray as xr
from utils import check_input, get_configs
import os
import pandas as pd
cfg = get_configs()

class OceanForcing:
    def __init__(self, model_dir):
        self.path = f"{model_dir}/1995-2100/"
        self.model = path.split('/')[-1]  # 3rd to last folder in directory structure
        
        # Load all data: thermal forcing, salinity, and temperature
        files = get_all_filepaths(path=self.path, filetype='nc')
        for file in files:
            if 'salinity' in file:
                self.salinity_data = xr.open_dataset(file)
            elif 'thermal_forcing' in file:
                self.thermal_forcing_data = xr.open_dataset(file)
            elif 'temperature' in file:
                self.temperature_data = xr.open_dataset(file)
            else:
                pass


    def aggregate_dims(self,):
        dims = self.data.dims
        if 'time' in dims:
            self.data = self.data.mean(dim='time')
        if 'nv4' in dims:
            self.data = self.data.mean(dim='nv4')
        return self

    def save_as_csv(self):
        if not isinstance(selt.data, pd.DataFrame):
            if self.datatype != "NetCDF":
                raise AttributeError(f'Data type must be \"NetCDF\", received {self.datatype}.')
                
            csv_path = f"{self.path[:-3]}.csv"
            self.data = self.data.to_dataframe()
        self.data.to_csv(csv_path)
        return self

    def add_sectors(self, grids):
        self.data = self.data.drop(labels=['lon_bnds', 'lat_bnds', 'lat2d', 'lon2d'])
        self.data = self.data.to_dataframe().reset_index(level='time', drop=True)
        self.data = pd.merge(self.data, grids.data, left_index=True, right_index=True)
        return self
