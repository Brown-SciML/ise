import xarray as xr
import xesmf as xe
import numpy as np
from ise.utils.io import check_type


def regrid_netcdf(source: str | xr.Dataset, target: str | xr.Dataset, method: str='patch', output_path: str=None,) -> xr.Dataset:
    """
    Regrid a source NetCDF dataset to match the grid of a target NetCDF dataset.
    Args:
        source (str | xr.Dataset): The source dataset or path to the source NetCDF file.
        target (str | xr.Dataset): The target dataset or path to the target NetCDF file.
        method (str): The regridding method to use. Options are 'bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s', 'auto'.
        output_path (str): The path to save the regridded dataset to. If None, the dataset is not saved.
    Returns:
        xr.Dataset: The regridded dataset.
    """

    # Validate input types
    check_type(source, (str, xr.Dataset,))
    check_type(target, (str, xr.Dataset))
    check_type(output_path, (str, type(None)))

    # Load source and target data
    if isinstance(source, str):
        source = xr.open_dataset(source)
    if isinstance(target, str):
        target = xr.open_dataset(target)
    
    # Create regridder
    regridder = xe.Regridder(source, target, method,)

    # Apply regridding
    ds_regridded = regridder(source)

    # Save to NetCDF
    if output_path:
        ds_regridded.to_netcdf(output_path)
    
    return ds_regridded

def add_sectors(ds: xr.Dataset, grids: xr.Dataset) -> xr.Dataset:
    """
    Add sector information to a dataset.
    Args:
        ds (xr.Dataset): The dataset to add sector information to.
        grids (xr.Dataset): The gridfile containing sector information.
    Returns:
        xr.Dataset: The dataset with sector information added.
    """
    check_type(ds, xr.Dataset)
    check_type(grids, xr.Dataset)
    # Check if the shapes of the datasets match
    if ds.sizes.mapping['x'] != grids.sizes.mapping['x'] or ds.sizes.mapping['y'] != grids.sizes.mapping['y']:
        raise ValueError("The dimensions of the dataset and the gridfile do not match.")
    
    ds = ds.assign(sectors=grids['sectors'])
    return ds

def interp_zeros(ds, var_name, method="linear"):
    """
    Replaces zero values with NaNs and interpolates missing values in an xarray dataset.

    Parameters:
    ds (xarray.Dataset): Input dataset with x, y, and time dimensions.
    var_name (str): The variable name to interpolate.
    method (str): Interpolation method ('linear', 'nearest', 'cubic', etc.)

    Returns:
    xarray.Dataset: Dataset with interpolated values.
    """
    check_type(ds, xr.Dataset)
    check_type(var_name, str)
    check_type(method, str)
    
    ds[var_name] = ds[var_name].where(ds[var_name] != 0, np.nan)  # Replace zeros with NaN
    ds[var_name] = ds[var_name].interpolate_na(dim="x", method=method)  # Interpolate over x
    
    return ds