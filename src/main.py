from data.processing.aggregate_by_sector import aggregate_all
from utils import get_all_filepaths
import xarray as xr
import pandas as pd

af_directory = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/"
aggregate_all(af_directory)