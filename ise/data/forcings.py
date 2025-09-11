import xarray as xr
from typing import List
import numpy as np
from ise.data.grids import GridFile
from ise.data.utils import convert_and_subset_times


class ForcingFile:
    def __init__(self, ice_sheet: str, realm: str, filepath: str, varname: str=None) -> None:
        self.ice_sheet = ice_sheet
        self.realm = realm
        self.filepath = filepath
        self.data = None
        self.sector_averages = None
        self.sectors = None
        self.varname = varname

    def load(self, filepath: str = None, validate=True, **kwargs) -> xr.Dataset:
        if filepath is None:
            filepath = self.filepath
        if self.data is not None:
            return self.data
        
        self.data = xr.open_dataset(filepath, **kwargs)
        if validate:
            self._validate_data()
        return self.data

    def drop_vars(self, vars: List[str],) -> xr.Dataset:
        # drop vars, flexible with try, except
        for var in vars:
            if var in self.data.dims:
                self.data = self.data.drop_dims(var)
            elif var in self.data.variables:
                self.data = self.data.drop_vars(var)

    def format_timestamps(self,) -> xr.Dataset:
        self.data = convert_and_subset_times(self.data)
        return self.data

    
    def get_data(self,) -> xr.Dataset:
        return self.data
    
    def aggregate_depth(self, method="mean"):
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
        if self.data is None:
            raise ValueError("No data loaded. Call load() before assigning sectors.")
        if isinstance(sectors, GridFile):
            sectors = sectors.get_sectors()
        self.sectors = sectors
        
        self.data['sector'] = sectors

        self._standardize_dims()
        return self.data

    
    def average_over_sector(self, sector_number: int=None) -> xr.Dataset:
        if self.data is None:
            raise ValueError("No data loaded. Call load() before averaging sectors.")
        if self.sectors is None and "sector" not in self.data and "sectors" not in self.data:
            raise ValueError("No sectors assigned. Call assign_sectors() before averaging sectors.")
        
        if self._check_averaged_sectors():
            return self.data
        
        if sector_number is None:
            raise NotImplementedError("Averaging over all sectors at once is not implemented.")
        mask = self.data["sector"] == sector_number
        sector_averages = self.data.where(mask, drop=True).mean(dim=['x', 'y'], skipna=True)

        self.sector_averages = sector_averages
        return sector_averages
    
    def _check_averaged_sectors(self):
        if tuple(self.data.dims) == ("sector", "time") or tuple(self.data.dims) == ("time", "sector"):
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