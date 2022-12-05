import time
import xarray as xr
from ise.utils.utils import get_configs, get_all_filepaths, check_input
import numpy as np
np.random.seed(10)
import pandas as pd
import time

def process_forcings(forcing_directory, export_directory, to_process='all', verbose=False):
    # check inputs
    to_process_options = ['all', 'atmosphere', 'ocean', 'ice_collapse']
    if isinstance(to_process, str):
        if to_process.lower() not in to_process_options:
            raise ValueError(f'to_process arg must be in [{to_process_options}], received {to_process}')
    elif isinstance(to_process, list):
        to_process_valid = all([(s in to_process_options) for s in to_process])
        if not to_process_valid:
            raise  ValueError(f'to_process arg must be in [{to_process_options}], received {to_process}')
        
        
    if to_process.lower() == 'all':
        to_process = ['atmosphere', 'ocean', 'ice_collapse']
    
    if verbose:
        print('Processing...')
    
    curr_time = time.time()
    if 'atmosphere' in to_process:
        af_directory = f"{forcing_directory}/Atmosphere_Forcing/"
        aggregate_atmosphere(af_directory, export=export_directory, )
        if verbose:
            prev_time, curr_time = curr_time, time.time()
            curr_time = time.time()
            print(f'Finished processing atmosphere, Total Running Time: {(curr_time - prev_time) // 60} minutes')
    
    if 'ocean' in to_process:
        of_directory = f"{forcing_directory}/Ocean_Forcing/"
        aggregate_ocean(of_directory, export=export_directory, )
        if verbose:
            prev_time, curr_time = curr_time, time.time()
            curr_time = time.time()
            print(f'Finished processing ocean, Total Running Time: {(curr_time - prev_time) // 60} minutes')
    
    if 'ice_collapse' in to_process:
        ice_directory = f"{forcing_directory}/Ice_Shelf_Fracture"
        aggregate_icecollapse(ice_directory, export=export_directory, )
        if verbose:
            prev_time, curr_time = curr_time, time.time()
            curr_time = time.time()
            print(f'Finished processing ice_collapse, Total Running Time: {(curr_time - prev_time) // 60} minutes')
    
    if verbose:
        print(f'Finished. Data exported to {export_directory}')
    
    

class AtmosphereForcing:
    def __init__(self, path):
        self.forcing_type = 'atmosphere'
        self.path = path
        self.aogcm = path.split('/')[-3]  # 3rd to last folder in directory structure
        
        if path[-2:] == 'nc':
            self.data = xr.open_dataset(self.path, decode_times=False)
            self.datatype = 'NetCDF'

        elif path[-3:] == 'csv':
            self.data = pd.read_csv(self.path,)
            self.datatype = 'CSV'


    def aggregate_dims(self,):
        dims = self.data.dims
        if 'time' in dims:
            self.data = self.data.mean(dim='time')
        if 'nv4' in dims:
            self.data = self.data.mean(dim='nv4')
        return self

    def save_as_csv(self):
        if not isinstance(self.data, pd.DataFrame):
            if self.datatype != "NetCDF":
                raise AttributeError(f'Data type must be \"NetCDF\", received {self.datatype}.')
                
            csv_path = f"{self.path[:-3]}.csv"
            self.data = self.data.to_dataframe()
        self.data.to_csv(csv_path)
        return self

    def add_sectors(self, grids):
        self.data = self.data.drop(labels=['lon_bnds', 'lat_bnds', 'lat2d', 'lon2d'])
        self.data = self.data.to_dataframe().reset_index(level='time', drop=True)
        self.data = pd.merge(self.data, grids.data, left_index=True, right_index=True, how='outer')
        return self



class OceanForcing:
    def __init__(self, aogcm_dir):
        self.forcing_type = 'ocean'
        self.path = f"{aogcm_dir}/1995-2100/"
        self.aogcm = aogcm_dir.split('/')[-2]  # 3rd to last folder in directory structure
        
        # Load all data: thermal forcing, salinity, and temperature
        files = get_all_filepaths(path=self.path, filetype='nc')
        for file in files:
            if 'salinity' in file:
                self.salinity_data = xr.open_dataset(file)
            elif 'thermal_forcing' in file:
                self.thermal_forcing_data = xr.open_dataset(file)
            elif 'temperature' in file:
                self.temperature_data = xr.open_dataset(file)
            else:
                pass


    def aggregate_dims(self,):
        dims = self.data.dims
        if 'z' in dims:
            self.data = self.data.mean(dim='time')
        if 'nbounds' in dims:
            self.data = self.data.mean(dim='nv4')
        return self

    def save_as_csv(self):
        if not isinstance(self.data, pd.DataFrame):
            if self.datatype != "NetCDF":
                raise AttributeError(f'Data type must be \"NetCDF\", received {self.datatype}.')
                
            csv_path = f"{self.path[:-3]}.csv"
            self.data = self.data.to_dataframe()
        self.data.to_csv(csv_path)
        return self

    def add_sectors(self, grids):      
        self.salinity_data = self.salinity_data.drop(labels=['z_bnds', 'lat', 'lon'])
        self.salinity_data = self.salinity_data.mean(dim='z', skipna=True).to_dataframe()
        self.salinity_data = self.salinity_data.reset_index(level='time',)
        self.salinity_data = pd.merge(self.salinity_data, grids.data, left_index=True, right_index=True, how='outer')
        self.salinity_data['year'] = self.salinity_data['time'].apply(lambda x: x.year)
        self.salinity_data = self.salinity_data.drop(columns=['time', 'mapping'])
        
        self.thermal_forcing_data = self.thermal_forcing_data.drop(labels=['z_bnds'])
        self.thermal_forcing_data = self.thermal_forcing_data.mean(dim='z', skipna=True).to_dataframe().reset_index(level='time',)
        self.thermal_forcing_data = pd.merge(self.thermal_forcing_data, grids.data, left_index=True, right_index=True, how='outer')
        self.thermal_forcing_data['year'] = self.thermal_forcing_data['time'].apply(lambda x: x.year)
        self.thermal_forcing_data = self.thermal_forcing_data.drop(columns=['time', 'mapping'])
        
        self.temperature_data = self.temperature_data.drop(labels=['z_bnds'])
        self.temperature_data = self.temperature_data.mean(dim='z', skipna=True).to_dataframe().reset_index(level='time',)
        self.temperature_data = pd.merge(self.temperature_data, grids.data, left_index=True, right_index=True, how='outer')
        self.temperature_data['year'] = self.temperature_data['time'].apply(lambda x: x.year)
        self.temperature_data = self.temperature_data.drop(columns=['time', 'mapping'])
        
        return self


class IceCollapse:
    def __init__(self, aogcm_dir):
        self.forcing_type = 'ice_collapse'
        self.path = f"{aogcm_dir}"
        self.aogcm = aogcm_dir.split('/')[-2]  # last folder in directory structure
        
        # Load all data: thermal forcing, salinity, and temperature
        files = get_all_filepaths(path=self.path, filetype='nc')
        
        if len(files) > 1: # if there is a "v2" file in the directory, use that one
            for file in files:
                if 'v2' in file:
                    self.data = xr.open_dataset(file)
                else:
                    pass
        else:
            self.data = xr.open_dataset(files[0])


    def save_as_csv(self):
        if not isinstance(self.data, pd.DataFrame):
            if self.datatype != "NetCDF":
                raise AttributeError(f'Data type must be \"NetCDF\", received {self.datatype}.')
                
            csv_path = f"{self.path[:-3]}.csv"
            self.data = self.data.to_dataframe()
        self.data.to_csv(csv_path)
        return self

    def add_sectors(self, grids):    
        self.data = self.data.drop(labels=['lon', 'lon_bnds', 'lat', 'lat_bnds'])
        self.data = self.data.to_dataframe().reset_index(level='time', drop=False)
        self.data = pd.merge(self.data, grids.data, left_index=True, right_index=True, how='outer')
        self.data['year'] = self.data['time'].apply(lambda x: x.year)
        self.data = self.data.drop(columns=['time', 'mapping', 'lat', 'lon',])
        return self
        




class GridSectors:
    def __init__(self, grid_size=8, filetype='nc', format_index=True):
        check_input(grid_size, [4, 8, 16, 32])
        check_input(filetype.lower(), ['nc', 'csv'])
        self.grids_dir = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/ISMIP6_sectors/"
        
        if filetype.lower() == 'nc':
            self.path = self.grids_dir + f"sectors_{grid_size}km.nc"
            self.data = xr.open_dataset(self.path, decode_times=False)
            self = self._to_dataframe()
            if format_index:
                self = self._format_index()
        elif filetype.lower() == 'csv':
            self.path = self.grids_dir + f"sector_{grid_size}.csv"
            self.data = pd.read_csv(self.path)
        else:
            raise NotImplementedError('Only \"NetCDF\" and \"CSV\" are currently supported')
    
    def _netcdf_to_csv(self):
        if self.filetype != "NetCDF":
            raise AttributeError(f'Data type must be \"NetCDF\", received {self.datatype}.')
            
        csv_path = f"{self.path[:-3]}.csv"
        df = self.data.to_dataframe()
        df.to_csv(csv_path)

    def _to_dataframe(self):
        if not isinstance(self, pd.DataFrame):
            self.data = self.data.to_dataframe()
        return self

    def _format_index(self):
        index_array = list(np.arange(0,761))
        self.data.index = pd.MultiIndex.from_product([index_array, index_array], names=['x', 'y'])
        return self

    
    


def aggregate_by_sector(path):
    """Takes a atmospheric forcing dataset, adds sector numbers to it,
    and gets aggregate data based on sector and year. Returns atmospheric
    forcing data object.

    Args:
        path (str): path to atmospheric forcing nc file

    Returns:
        Obj: AtmosphereForcing instance with aggregated data
    """
    # Load grid data with 8km grid size
    
    print('')

    # Load in Atmospheric forcing data and add the sector numbers to it
    if 'Atmosphere' in path:
        grids = GridSectors(grid_size=8,)
        forcing = AtmosphereForcing(path=path)
        
    elif 'Ocean' in path:
        grids = GridSectors(grid_size=8, format_index=False)
        forcing = OceanForcing(aogcm_dir=path)
        
    elif 'Ice' in path:
        grids = GridSectors(grid_size=8,)
        forcing = IceCollapse(path)

    forcing = forcing.add_sectors(grids)

    
    # Group the dataset and assign aogcm column to the aogcm simulation
    if forcing.forcing_type in ('atmosphere', 'ice_collapse'):
        forcing.data = forcing.data.groupby(['sectors', 'year']).mean()
        forcing.data['aogcm'] = forcing.aogcm.lower()
    elif forcing.forcing_type == 'ocean':
        forcing.salinity_data = forcing.salinity_data.groupby(['sectors', 'year']).mean()
        forcing.salinity_data['aogcm'] = forcing.aogcm.lower()
        forcing.temperature_data = forcing.temperature_data.groupby(['sectors', 'year']).mean()
        forcing.temperature_data['aogcm'] = forcing.aogcm.lower()
        forcing.thermal_forcing_data = forcing.thermal_forcing_data.groupby(['sectors', 'year']).mean()
        forcing.thermal_forcing_data['aogcm'] = forcing.aogcm.lower()
    
    return forcing



# TODO: Maybe make each of these aggregate functions a method?
def aggregate_atmosphere(directory, export, model_in_columns=False,):
    """Loops through every NC file in the provided forcing directory
    from 1995-2100 and applies the aggregate_by_sector function. It then outputs
    the concatenation of all processed data to all_data.csv 

    Args:
        directory (str): Directory containing forcing files
    """
    start_time = time.time()

    # Get all NC files that contain data from 1995-2100
    filepaths = get_all_filepaths(path=directory, filetype='nc')
    filepaths = [f for f in filepaths if "1995-2100" in f]
    

    # Useful progress prints
    print(f"Files to be processed...")
    print([f.split("/")[-1] for f in filepaths])

    # Loop over each file specified above
    all_data = pd.DataFrame()
    for i, fp in enumerate(filepaths):
        print('')
        print(f'File {i+1} / {len(filepaths)}')
        print(f'File: {fp.split("/")[-1]}')
        print(f'Time since start: {(time.time()-start_time) // 60} minutes')

        # attach the sector to the data and groupby sectors & year
        forcing = aggregate_by_sector(fp)

        # Handle files that don't have mrro_anomaly input (ISPL RCP 85?)
        try:
            forcing.data['mrro_anomaly']
        except KeyError:
            forcing.data['mrro_anomaly'] = np.nan

        # Keep selected columns and output each file individually
        forcing.data = forcing.data[['pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly', 'smb_anomaly', 'ts_anomaly', 'regions', 'aogcm',]]
    
        # forcing.data.to_csv(f"{fp[:-3]}_sectoryeargrouped.csv")

        # meanwhile, create a concatenated dataset
        all_data = pd.concat([all_data, forcing.data])
            
        print(' -- ')
    
    
    if model_in_columns:
        data = {'atmospheric_forcing': all_data}
        all_data = aogcm_to_features(data=data, export_dir=export)
    
    else:
        if export:
            all_data.to_csv(f"{export}/atmospheric_forcing.csv")
        
        
def aggregate_ocean(directory, export, model_in_columns=False, ):
    """Loops through every NC file in the provided forcing directory
    from 1995-2100 and applies the aggregate_by_sector function. It then outputs
    the concatenation of all processed data to all_data.csv 


    Args:
        directory (str): Import directory for oceanic forcing files (".../Ocean_Forcing/")
        export (str): Export directory to store output files
        model_in_columns (bool, optional): Wither to format AOGCM model as columns. Defaults to False.
    """
    start_time = time.time()

    # Get all NC files that contain data from 1995-2100
    filepaths = get_all_filepaths(path=directory, filetype='nc')
    filepaths = [f for f in filepaths if "1995-2100" in f]
    
    # In the case of ocean forcings, use the filepaths of the files to determine
    # which directories need to be used for OceanForcing processing. Change to
    # those directories rather than individual files.
    aogcms = list(set([f.split('/')[-3] for f in filepaths]))
    filepaths = [f"{directory}/{aogcm}/" for aogcm in aogcms]

    # Useful progress prints
    print(f"Files to be processed...")
    print([f.split("/")[-2] for f in filepaths])

    # Loop over each directory specified above
    salinity_data = pd.DataFrame()
    temperature_data = pd.DataFrame()
    thermal_forcing_data = pd.DataFrame()
    for i, fp in enumerate(filepaths):
        print('')
        print(f'Directory {i+1} / {len(filepaths)}')
        print(f'Directory: {fp.split("/")[-2]}')
        print(f'Time since start: {(time.time()-start_time) // 60} minutes')

        # attach the sector to the data and groupby sectors & year
        forcing = aggregate_by_sector(fp)

        forcing.salinity_data = forcing.salinity_data[['salinity', 'regions', 'aogcm']]
        forcing.temperature_data = forcing.temperature_data[['temperature', 'regions', 'aogcm']]
        forcing.thermal_forcing_data = forcing.thermal_forcing_data[['thermal_forcing', 'regions', 'aogcm']]
        
        
        # meanwhile, create a concatenated dataset
        salinity_data = pd.concat([salinity_data, forcing.salinity_data])
        temperature_data = pd.concat([temperature_data, forcing.temperature_data])
        thermal_forcing_data = pd.concat([thermal_forcing_data, forcing.thermal_forcing_data])
        
        # salinity_data.to_csv(export+'/_salinity.csv')
        # temperature_data.to_csv(export+'/_temperature.csv')
        # thermal_forcing_data.to_csv(export+'/_thermal_forcing.csv')
        
    print(' -- ')
    
    if model_in_columns:
        # For each concatenated dataset
        data = {'salinity': salinity_data, 'temperature': temperature_data, 'thermal_forcing': thermal_forcing_data}
        all_data = aogcm_to_features(data, export_dir=export)
    
    else:
        if export:
            salinity_data.to_csv(export+'/salinity.csv')
            temperature_data.to_csv(export+'/temperature.csv')
            thermal_forcing_data.to_csv(export+'/thermal_forcing.csv')
            
def aggregate_icecollapse(directory, export, model_in_columns=False, ):
    """Loops through every NC file in the provided forcing directory
    from 1995-2100 and applies the aggregate_by_sector function. It then outputs
    the concatenation of all processed data to all_data.csv 


    Args:
        directory (str): Import directory for oceanic forcing files (".../Ocean_Forcing/")
        export (str): Export directory to store output files
        model_in_columns (bool, optional): Wither to format AOGCM model as columns. Defaults to False.
    """
    start_time = time.time()

    # Get all NC files that contain data from 1995-2100
    filepaths = get_all_filepaths(path=directory, filetype='nc')
    
    # In the case of ocean forcings, use the filepaths of the files to determine
    # which directories need to be used for OceanForcing processing. Change to
    # those directories rather than individual files.
    aogcms = list(set([f.split('/')[-2] for f in filepaths]))
    filepaths = [f"{directory}/{aogcm}/" for aogcm in aogcms]

    # Useful progress prints
    print(f"Files to be processed...")
    print([f.split("/")[-2] for f in filepaths])

    # Loop over each directory specified above
    ice_collapse = pd.DataFrame()
    for i, fp in enumerate(filepaths):
        print('')
        print(f'Directory {i+1} / {len(filepaths)}')
        print(f'Directory: {fp.split("/")[-2]}')
        print(f'Time since start: {(time.time()-start_time) // 60} minutes')

        # attach the sector to the data and groupby sectors & year
        forcing = aggregate_by_sector(fp)

        forcing.data = forcing.data[['mask', 'regions', 'aogcm']]
        
        
        # meanwhile, create a concatenated dataset
        ice_collapse = pd.concat([ice_collapse, forcing.data])

        
    print(' -- ')
    
    if model_in_columns:
        # For each concatenated dataset
        data = {'ice_collapse': ice_collapse,}
        all_data = aogcm_to_features(data, export_dir=export)
    
    else:
        if export:
            ice_collapse.to_csv(export+'/ice_collapse.csv')

            
            
# ! Deprecated -- not useful
def aogcm_to_features(data: dict, export_dir: str):
        
    for key, all_data in data.items():
        separate_aogcm_dataframes = [y for x, y in all_data.groupby('aogcm')]
        
        # Change columns names in each dataframe
        for df in separate_aogcm_dataframes:
            aogcm = df.aogcm.iloc[0]
            df.columns = [f"{x}_{aogcm}" if x not in ['sectors', 'year', 'region', 'aogcm'] else x for x in df.columns ]
            
        # Merge dataframes together on common columns [sectors, year], resulting in 
        # one dataframe with sector, year, region, and columns for each aogcm variables
        all_data = separate_aogcm_dataframes[0]
        all_data = all_data.drop(columns=['aogcm'])
    
        for df in separate_aogcm_dataframes[1:]:
            df = df.drop(columns=['aogcm'])
            all_data = pd.merge(all_data, df, on=['sectors', 'year',], how='outer')
            
        region_cols = [c for c in all_data.columns if 'region' in c]
        non_region_cols = [c for c in all_data.columns if 'region' not in c]
        all_data = all_data[non_region_cols]
        
        # region assignment produces NA's, low priority -- do later
        # all_data['region'] = separate_aogcm_dataframes[0][region_cols[0]].reset_index(drop=True)
        all_data = all_data.drop_duplicates() # See why there are duplicates -- until then, this works
        
        if export_dir:
                all_data.to_csv(f"{export_dir}/{key}.csv")
            
    return all_data