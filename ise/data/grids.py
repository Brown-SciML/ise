import xarray as xr

class GridFile:

    def __init__(self, ice_sheet: str, filepath: str) -> None:
        self.ice_sheet = ice_sheet
        self.filepath = filepath
        self.data = None
        self.sector_variable_name = "sectors" if ice_sheet == "AIS" else "ID"

    def load(self, filepath: str = None, **kwargs) -> xr.Dataset:
        if filepath is None:
            filepath = self.filepath
        self.data = xr.open_dataset(filepath, **kwargs)
        return self.data

    def expand_dims(self, dim: str = "time", size: int = None) -> xr.Dataset:
        # expand out to N timestamps
        self.data = self.data.expand_dims({dim: size})
        return self.data
    
    def align_dims(self, dims: list = None) -> xr.Dataset:
        if dims is not None:
            self.data = self.data.transpose(*dims)
        else:
            self.data = self.data.transpose('time', 'x', 'y', ...)
        return self.data
    
    def get_sectors(self,) -> xr.DataArray:
        return self.data[self.sector_variable_name]
    
    def format_grids(self,) -> xr.Dataset:
        if self.data is None:
            self.load()
        self.expand_dims(size=86)
        self.align_dims()
        return self.data
    
    