from ise.grids.data.process import DimensionalityReducer, DatasetMerger, ProjectionProcessor
from ise.grids.utils import get_all_filepaths
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

forcing_directory = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS"
projection_directory = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-AIS/"
files = all_projection_fps = get_all_filepaths(path=projection_directory, filetype='nc', contains='ivaf', not_contains='ctrl_proj')
# all_forcing_fps = get_all_filepaths(path=forcing_directory, filetype='nc', contains='1995-2100', not_contains='Ice_Shelf_Fracture')
# all_forcing_fps = [x for x in all_forcing_fps if '8km' in x and 'v1' not in x]
# files.extend(all_forcing_fps)

for i, fp in tqdm(enumerate(files), total=len(files)):
    # fp = '/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/ccsm4_rcp2.6/Regridded_8km/CCSM4_8km_anomaly_rcp26_1995-2100.nc'
    # fp = '/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Ocean_Forcing/ukesm1-0-ll_ssp585/1995-2100/UKESM1-0-LL_ssp585_thermal_forcing_8km_x_60m.nc'
    # fp = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-AIS/DOE/MALI/exp13/ivaf_AIS_DOE_MALI_exp13.nc"
    
    if fp == '/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-AIS/NCAR/CISM/expD54/ivaf_AIS_NCAR_CISM_expD54.nc':
        continue
    d = xr.open_dataset(fp, decode_times=True)
    d = d.transpose('x', 'y', 'time', ...)
    
  
    
    ocean = False
    if 'Atmosphere' in fp:
        var = 'smb_anomaly'

    elif 'Ocean' in fp:
        ocean = True
        var = fp.split('/')[-1].split('_')[-4]
        if var == 'forcing':
            var = 'thermal_forcing'
    else:
        var = 'ivaf'
        
    if not ocean and ~np.isnan(d[var][:, :, 0]).any():
        values = d[var].values
        values[values == 0] = np.nan
        
    # on the first file, first iteration, keep the first 761*761 as the mask
    if i == 0:
        if ocean:
            mask = np.isnan(d[var][:, :, 0, 0]) * 1
        else:
            mask = np.isnan(d[var][:, :, 0]) * 1
   

    # loop through each sequential mask and only keep the minimum number of nan values
    for j in range(len(d.time)):
        # if j == 75:
        #     stop = ''
        if ocean:
            nan_index_new = np.isnan(d[var][:, :, j, 0]) * 1
        else:
            nan_index_new = np.isnan(d[var][:, :, j]) * 1
        # if an argument is np.nan for all data, the value of mask will stay 1, else 0
        mask = np.multiply(nan_index_new, mask)
        stop = ''
        
    
    if i % 10 == 0:
        plt.imshow(mask)
        plt.title(fp)
        
        plt.savefig(f'supplemental/nan_mask_pics/{fp.split("/")[-1].replace(".nc", "")}_{i}.png')
    del d # delete old dataset before loading next
    
plt.imshow(mask)
plt.title('Minimum np.nan')
plt.savefig('savefig.png')

# plt.savefig("savefig.png")
np.savetxt(f"nan_mask.csv", mask, delimiter=",")


