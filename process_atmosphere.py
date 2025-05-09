import argparse
import xarray as xr
import numpy as np
from ise.data.CMIP6.process import regrid_netcdf, add_sectors, interp_zeros
import seaborn as sns
import os

def create_single_nc_file(directory):
    single_nc = xr.open_mfdataset(f"{directory}/*.nc", combine="by_coords", use_cftime=True)
    return single_nc

def calculate_anomaly(dataset):
    climatology = dataset.sel(year=slice(1995, 2014)).mean('year', keep_attrs=True).compute()  # Forces computation once
    dataset = dataset.sel(year=slice(2015, 2100))
    anomaly = dataset - climatology  # Keeps Dask compatibility
    return climatology, anomaly


def process_atmosphere(output_dir, grid_file, historical_dir, ssp_dir):
    print("Loading data...")
    grids = xr.open_dataset(grid_file)
    historical = create_single_nc_file(historical_dir)
    ssp = create_single_nc_file(ssp_dir)
    
    print("Regridding data...")
    historical = regrid_netcdf(historical, grids, method='bilinear')
    ssp = regrid_netcdf(ssp, grids, method='bilinear')
    
    print("Interpolating zeros...")
    for var in ['pr', 'ts', 'evspsbl']:
        for nc in [historical, ssp]:
            nc = interp_zeros(nc, var)
    
    print("Grouping dataset into annual means...")
    dataset = xr.concat([historical, ssp], dim='time')
    dataset = dataset.groupby('time.year').mean('time')
    
    print('Calculating Climatology and Anomaly...')
    climatology, anomaly = calculate_anomaly(dataset)
    if not os.path.exists(f"{output_dir}/climatology.nc"):
        climatology.to_netcdf(f"{output_dir}/climatology.nc")
        print(f"Climatology dataset saved to {output_dir}/climatology.nc")
        
    anomaly = add_sectors(anomaly, grids)
    
    print('Calculating mean sector values...')
    mean_sector = anomaly.groupby('sectors').mean()
    
    print('Saving anomaly dataset...')
    mean_sector.to_netcdf(f"{output_dir}/anomaly.nc")
    
    print(f"Anomaly dataset saved to {output_dir}/anomaly.nc")
    print('Process complete.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CMIP6 atmosphere data and calculate anomalies.")
    
    parser.add_argument("output_dir", type=str, help="Directory to save the output NetCDF file.")
    parser.add_argument("grid_file", type=str, help="Path to the grid file (NetCDF format).")
    parser.add_argument("historical_dir", type=str, help="Directory containing historical NetCDF files.")
    parser.add_argument("ssp_dir", type=str, help="Directory containing SSP NetCDF files.")

    args = parser.parse_args()
    
    process_atmosphere(args.output_dir, args.grid_file, args.historical_dir, args.ssp_dir)
    # uv run process_atmosphere.py \
    #     /oscar/home/pvankatw/data/pvankatw/CMIP/access_cm2/ssp126/atmosphere \
    #     /oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Grid_Files/AIS_sectors_8km.nc \
    #     /oscar/home/pvankatw/data/pvankatw/CMIP/access_cm2/historical \
    #     /oscar/home/pvankatw/data/pvankatw/CMIP/access_cm2/ssp126/atmosphere
        
