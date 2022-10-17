import os
from data.classes.AtmosphereForcing import AtmosphereForcing
from utils import get_all_filepaths
import xarray as xr
import pandas as pd


# forcing = AtmosphereForcing(path=r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/ccsm4_rcp2.6/Regridded_8km/CCSM4_8km_anomaly_rcp26_1995-2100.nc")
# forcing = forcing.aggregate_dims()
# # print(forcing.data)
# forcing = forcing.save_as_df()
dir = "/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/"
dir = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/ISMIP6_sectors/"
files = get_all_filepaths(path=dir, filetype='csv')
# nc_files = [file for file in files if file.endswith('.nc')]

d = pd.read_csv(files[0])

# for i, nc in enumerate(nc_files):
#     print(f"Iteration: {i}, File: {nc}")
#     forcing = AtmosphereForcing(path=nc)
#     forcing = forcing.aggregate_dims()
#     forcing = forcing.save_as_df()


# forcing = AtmosphereForcing(path=r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/ccsm4_rcp2.6/Regridded_8km/CCSM4_8km_anomaly_rcp26_1995-2100.nc")
# forcing = forcing.aggregate_dims()
# print(forcing.data)

stop = ''

 