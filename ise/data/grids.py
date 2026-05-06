"""NetCDF sector-definition grid file loading and formatting.

``GridFile`` wraps the ice-sheet sector boundary grids used to assign each
spatial grid cell to a drainage sector (AIS: 18 sectors; GrIS: 6 drainage
basins).  The sector array it exposes is consumed by ``ForcingFile.assign_sectors()``
during the data processing pipeline.

Grid files expected
-------------------
AIS:
    ``AIS_sectors_8km.nc`` — sector variable named ``'sectors'``.
GrIS:
    ``GrIS_Basins_Rignot_sectors_5km.nc`` — sector variable named ``'ID'``.

Typical workflow
----------------
Sector grids need a time dimension that matches the forcing data (86 years)
before they can be broadcast alongside a forcing ``xarray.Dataset``.  The
``format_grids()`` convenience method handles the three required steps::

    from ise.data.grids import GridFile

    gridfile = GridFile("AIS", filepath="AIS_sectors_8km.nc")
    gridfile.format_grids()           # load → expand time to 86 → align dims
    sectors = gridfile.get_sectors()  # xr.DataArray of shape (time, x, y)

To perform steps individually (e.g. for a custom time length)::

    gridfile = GridFile("GrIS", filepath="GrIS_Basins_Rignot_sectors_5km.nc")
    gridfile.load()
    gridfile.expand_dims(dim="time", size=86)
    gridfile.align_dims(dims=["time", "x", "y"])
    sectors = gridfile.get_sectors()

In both cases the returned ``DataArray`` is passed directly to
``ForcingFile.assign_sectors(gridfile)`` or used as a mask in the sector-level
aggregation functions in ``ise.data.process``.
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
        self.data: xr.Dataset | None = None
        self.sector_variable_name = "sectors" if ice_sheet == "AIS" else "ID"

    def load(self, filepath: str | None = None, **kwargs) -> xr.Dataset:
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

    def expand_dims(self, dim: str = "time", size: int | None = None) -> xr.Dataset:
        """
        Expand dimensions (e.g. add time dimension of given size).

        Args:
            dim (str, optional): Dimension name. Defaults to 'time'.
            size (int, optional): Size of the new dimension. Defaults to None.

        Returns:
            xarray.Dataset: The dataset with expanded dimension.
        """
        assert self.data is not None, "No data loaded. Call load() first."
        self.data = self.data.expand_dims({dim: size})
        return self.data

    def align_dims(self, dims: list | None = None) -> xr.Dataset:
        """
        Transpose dimensions to a standard order.

        Args:
            dims (list, optional): Dimension order. If None, uses ('time', 'x', 'y', ...).

        Returns:
            xarray.Dataset: The dataset with reordered dimensions.
        """
        assert self.data is not None, "No data loaded. Call load() first."
        if dims is not None:
            self.data = self.data.transpose(*dims)
        else:
            self.data = self.data.transpose("time", "x", "y", ...)
        return self.data

    def get_sectors(
        self,
    ) -> xr.DataArray:
        """Return the sector ID array from the grid dataset."""
        assert self.data is not None, "No data loaded. Call load() first."
        return self.data[self.sector_variable_name]

    def format_grids(
        self,
    ) -> xr.Dataset:
        """
        Load (if needed), expand time to 86, and align dimensions.

        Returns:
            xarray.Dataset: The formatted grid dataset.
        """
        if self.data is None:
            self.load()
        self.expand_dims(size=86)
        self.align_dims()
        assert self.data is not None
        return self.data
