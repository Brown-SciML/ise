import xarray as xr
from utils import get_configs, get_all_filepaths
import pandas as pd
cfg = get_configs()

class IceCollapse:
    def __init__(self, model_dir):
        self.forcing_type = 'ice_collapse'
        self.path = f"{model_dir}"
        self.model = model_dir.split('/')[-2]  # last folder in directory structure
        
        # Load all data: thermal forcing, salinity, and temperature
        files = get_all_filepaths(path=self.path, filetype='nc')
        
        if len(files) > 1: # if there is a "v2" file in the directory, use that one
            for file in files:
                if 'v2' in files:
                    self.data = xr.open_dataset(file)
                else:
                    pass


    def save_as_csv(self):
        if not isinstance(self.data, pd.DataFrame):
            if self.datatype != "NetCDF":
                raise AttributeError(f'Data type must be \"NetCDF\", received {self.datatype}.')
                
            csv_path = f"{self.path[:-3]}.csv"
            self.data = self.data.to_dataframe()
        self.data.to_csv(csv_path)
        return self

    def add_sectors(self, grids):      
        self.data = self.data.drop(labels=['lon', 'lon_bnds', 'lat', 'lat_bnds'])
        self.data = self.data.to_dataframe().reset_index(level='time', drop=True)
        self.data = pd.merge(self.data, grids.data, left_index=True, right_index=True)
        self.data['year'] = self.data['time'].apply(lambda x: x.year)
        self.data = self.data.drop(columns=['time', 'mapping'])
        return self
        