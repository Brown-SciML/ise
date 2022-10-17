import xarray as xr
from utils import check_input, get_configs
import os
import pandas as pd
cfg = get_configs()

class AtmosphereForcing:
    def __init__(self, path):
        self.path = path
        
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

        if self.datatype != "NetCDF":
            raise AttributeError(f'Data type must be \"NetCDF\", received {self.datatype}.')
            
        csv_path = f"{self.path[:-3]}.csv"
        df = self.data.to_dataframe()
        df.to_csv(csv_path)
        return self

    def _add_sectors


# forcing = AtmosphereForcing(simulation='ccsm4_rcp2.6', time_frame='1995-2100')
# forcing = AtmosphereForcing(path=r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/ccsm4_rcp2.6/Regridded_8km/CCSM4_8km_anomaly_rcp26_1995-2100.nc")
# print(forcing.data)
# stop = []