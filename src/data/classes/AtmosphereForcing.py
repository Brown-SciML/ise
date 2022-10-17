import xarray as xr
from utils import check_input, get_configs
import os
import pandas as pd
cfg = get_configs()

class AtmosphereForcing:
    def __init__(self, path):
        self.path = path
        self.model = path.split('/')[-3]  # 3rd to last folder in directory structure
        
        if path[-2:] == 'nc':
            self.data = xr.open_dataset(self.path, decode_times=False)
            self.datatype = 'NetCDF'

        elif path[-3:] == 'csv':
            self.data = pd.read_csv(self.path,)
            self.datatype = 'CSV'


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

    def add_sectors(self, GridSectors):
        self.data = self.data.drop(labels=['lon_bnds', 'lat_bnds', 'lat2d', 'lon2d'])
        self.data = self.data.to_dataframe().reset_index(level='time', drop=True)
        self.data = pd.merge(self.data, GridSectors.data, left_index=True, right_index=True)
        return self

    def group(self, columns):
        self.data = self.data.groupby(columns).mean()
        return self
