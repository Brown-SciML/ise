import xarray as xr
from utils import get_configs, get_all_filepaths
import pandas as pd
cfg = get_configs()

class OceanForcing:
    def __init__(self, model_dir):
        self.forcing_type = 'ocean'
        self.path = f"{model_dir}/1995-2100/"
        self.model = model_dir.split('/')[-1]  # 3rd to last folder in directory structure
        
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
        if 'z' in dims:
            self.data = self.data.mean(dim='time')
        if 'nbounds' in dims:
            self.data = self.data.mean(dim='nv4')
        return self

    def save_as_csv(self):
        if not isinstance(self.data, pd.DataFrame):
            if self.datatype != "NetCDF":
                raise AttributeError(f'Data type must be \"NetCDF\", received {self.datatype}.')
                
            csv_path = f"{self.path[:-3]}.csv"
            self.data = self.data.to_dataframe()
        self.data.to_csv(csv_path)
        return self

    def add_sectors(self, grids):      
        self.salinity_data = self.salinity_data.drop(labels=['z_bnds'])
        self.salinity_data = self.salinity_data.mean(dim='z', skipna=True).to_dataframe().reset_index(level='time',)
        self.salinity_data = pd.merge(self.salinity_data, grids.data, left_index=True, right_index=True)
        self.salinity_data['year'] = self.salinity_data['time'].apply(lambda x: x.year)
        self.salinity_data = self.salinity_data.drop(columns=['time'])
        
        self.thermal_forcing_data = self.thermal_forcing_data.drop(labels=['z_bnds'])
        self.thermal_forcing_data = self.thermal_forcing_data.mean(dim='z', skipna=True).to_dataframe().reset_index(level='time',)
        self.thermal_forcing_data = pd.merge(self.thermal_forcing_data, grids.data, left_index=True, right_index=True)
        self.thermal_forcing_data['year'] = self.thermal_forcing_data['time'].apply(lambda x: x.year)
        self.thermal_forcing_data = self.thermal_forcing_data.drop(columns=['time'])
        
        self.temperature_data = self.temperature_data.drop(labels=['z_bnds'])
        self.temperature_data = self.temperature_data.mean(dim='z', skipna=True).to_dataframe().reset_index(level='time',)
        self.temperature_data = pd.merge(self.temperature_data, grids.data, left_index=True, right_index=True)
        self.temperature_data['year'] = self.temperature_data['time'].apply(lambda x: x.year)
        self.temperature_data = self.temperature_data.drop(columns=['time'])
        
        return self
