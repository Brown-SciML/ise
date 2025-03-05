import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load CMIP6 source dataset
data_directory = r"/oscar/home/pvankatw/data/pvankatw/CMIP/"
data_path = r"pr_day_MPI-ESM1-2-HR_ssp585_r1i1p1f1_gn_20150101-20191231.nc"
source = xr.open_dataset(data_directory + data_path)

# Load regridded dataset
regridded_path = "bilinear.nc"
ds_regridded = xr.open_dataset(regridded_path)
ds_regridded = ds_regridded.rename({'__xarray_dataarray_variable__': 'pr'})

# Select a time slice (assuming the dataset has a time dimension)
time_index = 100

# Define Antarctica projection
projection = ccrs.SouthPolarStereo()

# Explicitly select the precipitation variable 'pr' before plotting
source_pr = source['pr'].isel(time=time_index)  # Select 'pr' from source dataset
regridded_pr = ds_regridded['pr'].isel(time=time_index)  # Select 'pr' from regridded dataset

# Compute shared color scale
vmin = min(source_pr.min(), regridded_pr.min())
vmax = max(source_pr.max(), regridded_pr.max())

# Now proceed with plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), subplot_kw={'projection': projection})

# Define extent for Antarctica Ice Sheet in EPSG:3031 projection
antarctic_extent = [-2.5e6, 2.5e6, -2.5e6, 2.5e6]  # X and Y limits in meters

# Plot original CMIP6 precipitation (assuming it is in lat/lon)
axes[0].set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
axes[0].set_title("Original CMIP6 Precipitation")
img1 = source_pr.plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax, add_colorbar=False)
axes[0].add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black')
axes[0].add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')

# Plot regridded precipitation (already in EPSG:3031, no transform needed)
axes[1].set_xlim(antarctic_extent[0], antarctic_extent[1])
axes[1].set_ylim(antarctic_extent[2], antarctic_extent[3])
axes[1].set_title("Regridded Precipitation (EPSG:3031)")
img2 = regridded_pr.plot(ax=axes[1], cmap='coolwarm', vmin=vmin, vmax=vmax, add_colorbar=False)
axes[1].add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black')
axes[1].add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')

# Add a single colorbar for both plots
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Adjust colorbar position
cbar = fig.colorbar(img1, cax=cbar_ax)
cbar.set_label("Precipitation (units)")  # Adjust label as necessary

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
plt.savefig('ais_regridded.png')
plt.show()
