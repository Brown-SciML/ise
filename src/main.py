import os
from data.AtmosphereForcing import AtmosphereForcing
from utils import get_all_filepaths
import xarray as xr


# forcing = AtmosphereForcing(path=r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/ccsm4_rcp2.6/Regridded_8km/CCSM4_8km_anomaly_rcp26_1995-2100.nc")
# forcing = forcing.aggregate_dims()
# # print(forcing.data)
# forcing = forcing.save_as_df()

# files = get_all_filepaths("/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/")
# nc_files = [file for file in files if file[-2:] == 'nc']

# for i, nc in enumerate(nc_files):
#     print(f"Iteration: {i}, File: {nc}")
#     forcing = AtmosphereForcing(path=nc)
#     forcing = forcing.aggregate_dims()
#     forcing = forcing.save_as_df()


forcing = AtmosphereForcing(path=r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/ccsm4_rcp2.6/Regridded_8km/CCSM4_8km_anomaly_rcp26_1995-2100.nc")
forcing = forcing.aggregate_dims()
print(forcing.data)

stop = ''

 