import os
from data.classes.AtmosphereForcing import AtmosphereForcing
from data.classes.GridSectors import GridSectors
from utils import get_all_filepaths
import xarray as xr
import pandas as pd

grids = GridSectors(grid_size=8)
atmosphere_forcing_path = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/ccsm4_rcp2.6/Regridded_8km/CCSM4_8km_anomaly_rcp26_1995-2100.nc"
atmosphere = AtmosphereForcing(path=atmosphere_forcing_path)
atmosphere = atmosphere.add_sectors(GridSectors)
atmosphere = atmosphere.group(columns=['sectors', 'year'])
print(atmosphere.data)

stop = ''

 