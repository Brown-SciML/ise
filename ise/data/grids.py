"""Grid file handling for sector definitions in ice sheet emulation.

This module provides the GridFile class for loading and formatting NetCDF
grid files that define sector boundaries (e.g. AIS 18 sectors, GrIS 6 regions).
"""
import xarray as xr


class GridFile:
    """
    Wrapper for loading and formatting sector grid NetCDF files.

    Used to load sector IDs and optionally expand/align dimensions for
    compatibility with forcing data (e.g. time dimension of length 86).

    Args:
        ice_sheet (str): Ice sheet identifier ('AIS' or 'GrIS').
        filepath (str): Path to the grid NetCDF file.

    Attributes:
        ice_sheet (str): Ice sheet identifier.
        filepath (str): Path to the file.
        data (xarray.Dataset or None): Loaded dataset after load().
        sector_variable_name (str): Name of the sector variable ('sectors' for AIS, 'ID' for GrIS).
    """

    def __init__(self, ice_sheet: str, filepath: str) -> None:
        self.ice_sheet = ice_sheet
        self.filepath = filepath
        self.data = None
        self.sector_variable_name = "sectors" if ice_sheet == "AIS" else "ID"

    def load(self, filepath: str = None, **kwargs) -> xr.Dataset:
        """
        Load the grid dataset from the NetCDF file.

        Args:
            filepath (str, optional): Override path. Defaults to self.filepath.
            **kwargs: Passed to xarray.open_dataset.

        Returns:
            xarray.Dataset: The loaded dataset.
        """
        if filepath is None:
            filepath = self.filepath
        self.data = xr.open_dataset(filepath, **kwargs)
        return self.data

    def expand_dims(self, dim: str = "time", size: int = None) -> xr.Dataset:
        """
        Expand dimensions (e.g. add time dimension of given size).

        Args:
            dim (str, optional): Dimension name. Defaults to 'time'.
            size (int, optional): Size of the new dimension. Defaults to None.

        Returns:
            xarray.Dataset: The dataset with expanded dimension.
        """
        self.data = self.data.expand_dims({dim: size})
        return self.data
    
    def align_dims(self, dims: list = None) -> xr.Dataset:
        """
        Transpose dimensions to a standard order.

        Args:
            dims (list, optional): Dimension order. If None, uses ('time', 'x', 'y', ...).

        Returns:
            xarray.Dataset: The dataset with reordered dimensions.
        """
        if dims is not None:
            self.data = self.data.transpose(*dims)
        else:
            self.data = self.data.transpose('time', 'x', 'y', ...)
        return self.data
    
    def get_sectors(self,) -> xr.DataArray:
        """Return the sector ID array from the grid dataset."""
        return self.data[self.sector_variable_name]

    def format_grids(self,) -> xr.Dataset:
        """
        Load (if needed), expand time to 86, and align dimensions.

        Returns:
            xarray.Dataset: The formatted grid dataset.
        """
        if self.data is None:
            self.load()
        self.expand_dims(size=86)
        self.align_dims()
        return self.data
    
    