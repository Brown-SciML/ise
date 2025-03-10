import xarray as xr
import xesmf as xe

from ise.utils.io import check_type


def regrid_netcdf(source: str | xr.Dataset, target: str | xr.Dataset, output_path: str=None,) -> xr.Dataset:
    """
    Regrid a source NetCDF dataset to match the grid of a target NetCDF dataset.
    Args:
        source (str | xr.Dataset): The source dataset or path to the source NetCDF file.
        target (str | xr.Dataset): The target dataset or path to the target NetCDF file.
    Returns:
        xr.Dataset: The regridded dataset.
    """

    # Validate input types
    check_type(source, (str, xr.Dataset))
    check_type(target, (str, xr.Dataset))
    check_type(output_path, (str, type(None)))

    # Load source and target data
    if isinstance(source, str):
        source = xr.open_dataset(source)
    if isinstance(target, str):
        target = xr.open_dataset(target)
    
    # Create regridder
    regridder = xe.Regridder(source, target, 'patch',)

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
    
    # Check if the shapes of the datasets match
    if ds.sizes.mapping['x'] != grids.sizes.mapping['x'] or ds.sizes.mapping['y'] != grids.sizes.mapping['y']:
        raise ValueError("The dimensions of the dataset and the gridfile do not match.")
    
    ds = ds.assign(sectors=grids['sectors'])
    return ds