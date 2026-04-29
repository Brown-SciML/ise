"""Climate forcing file handling for ice sheet emulation.

This module provides the ForcingFile class for loading, validating, and
processing NetCDF forcing data (atmospheric and oceanic) with sector-level
aggregation.
"""

from typing import List

import numpy as np
import xarray as xr

from ise.data.grids import GridFile
from ise.data.utils import convert_and_subset_times


class ForcingFile:
    """
    Wrapper for loading and processing climate forcing NetCDF files.

    Supports atmospheric and oceanic realms, sector assignment, depth aggregation
    (ocean), and sector-averaged time series.

    Args:
        ice_sheet (str): Ice sheet identifier ('AIS' or 'GrIS').
        realm (str): Forcing realm ('atmos' or 'ocean').
        filepath (str): Path to the NetCDF forcing file.
        varname (str, optional): Name of the data variable. Defaults to None (first data var).

    Attributes:
        ice_sheet (str): Ice sheet identifier.
        realm (str): Forcing realm.
        filepath (str): Path to the file.
        data (xarray.Dataset or None): Loaded dataset after load().
        sector_averages (xarray.Dataset or None): Sector-averaged data after average_over_sector().
        sectors (numpy.ndarray or None): Sector IDs after assign_sectors().
        varname (str or None): Data variable name.
    """

    def __init__(self, ice_sheet: str, realm: str, filepath: str, varname: str = None) -> None:
        self.ice_sheet = ice_sheet
        self.realm = realm
        self.filepath = filepath
        self.data = None
        self.sector_averages = None
        self.sectors = None
        self.varname = varname

    def load(self, filepath: str = None, validate=True, **kwargs) -> xr.Dataset:
        """
        Load the forcing dataset from the NetCDF file.

        Args:
            filepath (str, optional): Override path. Defaults to self.filepath.
            validate (bool, optional): Whether to validate (non-NaN data). Defaults to True.
            **kwargs: Passed to xarray.open_dataset.

        Returns:
            xarray.Dataset: The loaded dataset.
        """
        if filepath is None:
            filepath = self.filepath
        if self.data is not None:
            return self.data

        self.data = xr.open_dataset(filepath, **kwargs)
        if validate:
            self._validate_data()
        return self.data

    def drop_vars(
        self,
        vars: List[str],
    ) -> xr.Dataset:
        """
        Drop dimensions or variables from the loaded dataset.

        Args:
            vars (List[str]): Names of dimensions or variables to drop.

        Returns:
            xarray.Dataset: The dataset (modified in place).
        """
        for var in vars:
            if var in self.data.dims:
                self.data = self.data.drop_dims(var)
            elif var in self.data.variables:
                self.data = self.data.drop_vars(var)

    def format_timestamps(
        self,
    ) -> xr.Dataset:
        """
        Convert and subset time coordinate to 2015–2100 (86 years).

        Returns:
            xarray.Dataset: The dataset with formatted time.
        """
        self.data = convert_and_subset_times(self.data)
        return self.data

    def get_data(
        self,
    ) -> xr.Dataset:
        """Return the loaded dataset."""
        return self.data

    def aggregate_depth(self, method="mean"):
        """
        Aggregate over the depth dimension (ocean realm only).

        Args:
            method (str): 'mean' or 'sum'. Defaults to 'mean'.

        Returns:
            xarray.Dataset: The dataset with depth aggregated.

        Raises:
            ValueError: If realm is not 'ocean', data not loaded, or no 'z' dimension.
        """
        if self.realm != "ocean":
            raise ValueError("Depth aggregation is only applicable for ocean realm.")
        if self.data is None:
            raise ValueError("No data loaded. Call load() before aggregating depth.")
        if "z" not in self.data.dims:
            raise ValueError("Dataset has no 'z' dimension to aggregate over.")
        if method == "mean":
            self.data = self.data.mean(dim="z", skipna=True)
        elif method == "sum":
            self.data = self.data.sum(dim="z", skipna=True)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        return self.data

    def assign_sectors(self, sectors: np.ndarray | GridFile) -> xr.Dataset:
        """
        Assign sector IDs to the dataset (e.g. from a GridFile).

        Args:
            sectors (numpy.ndarray or GridFile): Sector IDs or GridFile to get sectors from.

        Returns:
            xarray.Dataset: The dataset with sector coordinate.

        Raises:
            ValueError: If data is not loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() before assigning sectors.")
        if isinstance(sectors, GridFile):
            sectors = sectors.get_sectors()
        self.sectors = sectors

        self.data["sector"] = sectors

        self._standardize_dims()
        return self.data

    def average_over_sector(self, sector_number: int = None) -> xr.Dataset:
        """
        Average data over grid cells within a sector (or all sectors).

        Args:
            sector_number (int, optional): Sector ID. If None, must be pre-averaged. Defaults to None.

        Returns:
            xarray.Dataset: Sector-averaged data.

        Raises:
            ValueError: If data not loaded or sectors not assigned.
            NotImplementedError: If sector_number is None (averaging all sectors at once).
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() before averaging sectors.")
        if self.sectors is None and "sector" not in self.data and "sectors" not in self.data:
            raise ValueError("No sectors assigned. Call assign_sectors() before averaging sectors.")

        if self._check_averaged_sectors():
            return self.data

        if sector_number is None:
            raise NotImplementedError("Averaging over all sectors at once is not implemented.")
        mask = self.data["sector"] == sector_number
        sector_averages = self.data.where(mask, drop=True).mean(dim=["x", "y"], skipna=True)

        self.sector_averages = sector_averages
        return sector_averages

    def _check_averaged_sectors(self):
        """Return True if data is already sector-averaged (dims sector, time)."""
        if tuple(self.data.dims) == ("sector", "time") or tuple(self.data.dims) == (
            "time",
            "sector",
        ):
            self.sector_averages = self.data
            return True
        else:
            return False

    def _standardize_dims(self):
        if "sectors" in self.data.dims:
            self.data = self.data.rename_dims({"sectors": "sector"})
        if "sectors" in self.data.coords:
            self.data = self.data.rename({"sectors": "sector"})

        var_name = list(self.data.data_vars)[0] if not self.varname else self.varname
        if self.data["sector"].dims != self.data[var_name].dims:
            # print(f"Transposing sector from {self.data['sector'].dims} to {self.data[var_name].dims}")
            self.data["sector"] = self.data["sector"].transpose(*self.data[var_name].dims)

        return self.data

    def _validate_data(self):
        self.var_name = list(self.data.data_vars)[0] if not self.varname else self.varname
        if self.data[self.var_name].isnull().all():
            raise ValueError(f"All values in variable {self.var_name} are NaN.")
        return True
