import xarray as xr
import os
import pandas as pd
import numpy as np
import warnings
import cftime
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle as pkl
from ise.grids.utils import get_all_filepaths
from datetime import date, timedelta, datetime
from tqdm import tqdm
import matplotlib.pyplot as plt



class ProjectionProcessor:
    """
    A class for processing ice sheet data.
    
    Attributes:
    - ice_sheet (str): Ice sheet to be processed. Must be 'AIS' or 'GIS'.
    - forcings_directory (str): The path to the directory containing the forcings data.
    - projections_directory (str): The path to the directory containing the projections data.
    - scalefac_path (str): The path to the netCDF file containing scaling factors for each grid cell.
    - densities_path (str): The path to the CSV file containing ice and ocean density (rhow/rhoi) data for each experiment.
    
    Methods:
    - __init__(self, ice_sheet, forcings_directory, projections_directory, scalefac_path=None, densities_path=None): Initializes the Processor object.
    - process_forcings(self): Processes the forcings data.
    - process_projections(self, output_directory): Processes the projections data.
    - _calculate_ivaf_minus_control(self, data_directory, densities_fp, scalefac_path): Calculates the ice volume above flotation (IVAF) for each file in the given data directory, subtracting out the control projection IVAF if applicable.
    - _calculate_ivaf_single_file(self, directory, densities, scalefac_model, ctrl_proj=False): Calculates the ice volume above flotation (IVAF) for a single file.
    """
    def __init__(self, ice_sheet, forcings_directory, projections_directory, scalefac_path=None, densities_path=None):
        self.forcings_directory = forcings_directory
        self.projections_directory = projections_directory
        self.densities_path = densities_path
        self.scalefac_path = scalefac_path
        self.ice_sheet = ice_sheet.upper()
        if self.ice_sheet.lower() in ('gris', 'gis'):
            self.ice_sheet = 'GIS'
        self.resolution = 5 if self.ice_sheet == 'GIS' else 8
    
    def process(self, ):
        """
        Process the ISMIP6 projections by calculating IVAF for both control 
        and experiments, subtracting out the control IVAF from experiments,
        and exporting ivaf files.
            
        Args:
            output_directory (str): The directory to save the processed projections.
        
        Raises:
            ValueError: If projections_directory or output_directory is not specified.
        
        Returns:
            int: 1 indicating successful processing.
        """
        if self.projections_directory is None:
            raise ValueError("Projections path must be specified")
        
        # if the last ivaf file is missing, assume none of them are and calculate and export all ivaf files
        if self.ice_sheet == "AIS": # and not os.path.exists(f"{self.projections_directory}/VUW/PISM/exp08/ivaf_GIS_VUW_PISM_exp08.nc"):
            self._calculate_ivaf_minus_control(self.projections_directory, self.densities_path, self.scalefac_path)
        elif self.ice_sheet == 'GIS': # and not os.path.exists(f"{self.projections_directory}/VUW/PISM/exp04/ivaf_AIS_VUW_PISM_exp04.nc"):
            self._calculate_ivaf_minus_control(self.projections_directory, self.densities_path, self.scalefac_path)
        
            
        return 1
 
    def _calculate_ivaf_minus_control(self, data_directory: str, densities_fp: str, scalefac_path: str):
        """
        Calculates the ice volume above flotation (IVAF) for each file in the given data directory, 
        subtracting out the control projection IVAF if applicable. 
        
        Args:
        - data_directory (str): path to directory containing the data files to process
        - densities_fp (str or pd.DataFrame): filepath to CSV file containing density data, or a pandas DataFrame
        - scalefac_path (str): path to netCDF file containing scaling factors for each grid cell
        
        Returns:
        - int: 1 indicating successful calculation.
        
        Raises:
        - ValueError: if densities_fp is None or not a string or pandas DataFrame
        
        """
        
        # error handling for densities argument (must be str filepath or dataframe)
        if densities_fp is None:
            raise ValueError("densities_fp must be specified. Run get_model_densities() to get density data.")
        if isinstance(densities_fp, str):
            densities = pd.read_csv(densities_fp)
        elif isinstance(densities_fp, pd.DataFrame):
            pass
        else:
            raise ValueError("densities argument must be a string or a pandas DataFrame.")
        
        # open scaling model 
        scalefac_model = xr.open_dataset(scalefac_path)
        scalefac_model = np.transpose(scalefac_model.af2.values, (1,0))
        
        # adjust scaling model based on desired resolution
        if self.ice_sheet == "AIS":
            scalefac_model = scalefac_model[::self.resolution, ::self.resolution]
        elif self.ice_sheet == 'GIS' and scalefac_model.shape != (337, 577):
            if scalefac_model.shape[0] == 6081:
                raise ValueError(f"Scalefac model must be 337x577 for GIS, received {scalefac_model.shape}. Make sure you are using the GIS scaling model and not the AIS.")
            raise ValueError(f"Scalefac model must be 337x577 for GIS, received {scalefac_model.shape}.")
        
        # get all files in directory with "ctrl_proj" and "exp" in them and store separately
        ctrl_proj_dirs = []
        exp_dirs = []
        for root, dirs, _ in os.walk(data_directory):
            for directory in dirs:
                if "ctrl_proj" in directory:
                    ctrl_proj_dirs.append(os.path.join(root, directory))
                elif 'exp' in directory:
                    exp_dirs.append(os.path.join(root, directory))
                else:
                    pass
        



        # first calculate ivaf for control projections
        for directory in ctrl_proj_dirs:
            self._calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=True)
            
        # then, for each experiment, calculate ivaf and subtract out control
        # exp_dirs = exp_dirs[65:]
        for directory in exp_dirs:
            self._calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=False)
            
            
        return 1
        
    def _calculate_ivaf_single_file(self, directory, densities, scalefac_model, ctrl_proj=False):  
        """
        Calculate the Ice Volume Above Floatation (IVAF) for a single file.

        Args:
            directory (str): The directory path of the file.
            densities (pandas.DataFrame): A DataFrame containing density values for different groups and models.
            scalefac_model (float): The scale factor for the model.
            ctrl_proj (bool, optional): Flag indicating whether the projection is a control projection. Defaults to False.

        Returns:
            int: 1 if the processing is successful, -1 otherwise.
            
            
        """

        # directory = r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-GrIS/AWI/ISSM1/exp09"
        # get metadata from path

        path = directory.split('/')
        exp = path[-1]
        model = path[-2]
        group = path[-3]
                
        # Determine which control to use based on experiment (only applies to AIS) per Nowicki, 2020
        if not ctrl_proj:
            if self.ice_sheet == 'AIS':
                if exp in ('exp01', 'exp02', 'exp03','exp04','exp11','expA1','expA2','expA3','expA4', 'expB1', 'expB2', 'expB3', 'expB4', 'expB5', 'expC2', 'expC5', 'expC8', 'expC11', 'expE1', 'expE2', 'expE3', 'expE4', 'expE5', 'expE11', 'expE12', 'expE13', 'expE14'):
                    ctrl_path = os.path.join("/".join(path[:-1]), f'ctrl_proj_open/ivaf_{self.ice_sheet}_{group}_{model}_ctrl_proj_open.nc')
                elif exp in ('exp05','exp06','exp07','exp08','exp09','exp10','exp12','exp13','expA5','expA6','expA7','expA8', 'expB6', 'expB7', 'expB8', 'expB9', 'expB10', 'expC3', 'expC6', 'expC9', 'expC12', 'expE6', 'expE7', 'expE8', 'expE9', 'expE10', 'expE15', 'expE16', 'expE17', 'expE18') or 'expD' in exp:
                    ctrl_path = os.path.join("/".join(path[:-1]), f'ctrl_proj_std/ivaf_{self.ice_sheet}_{group}_{model}_ctrl_proj_std.nc')
                elif exp in ('expC1', 'expC4', 'expC7', 'expC10'):  # N/A value for ocean_forcing in Nowicki, 2020 table A2
                    return -1
                else:
                    print(f"Experiment {exp} not recognized. Skipped.")
                    return -1
                    
            else: 
                # GrIS doesn't have ctrl_proj_open vs ctrl_proj_std
                ctrl_path = os.path.join("/".join(path[:-1]), f'ctrl_proj/ivaf_{self.ice_sheet}_{group}_{model}_ctrl_proj.nc')
            
            # for some reason there is no ctrl_proj_open for AWI and JPL1, skip
            if group == 'AWI'and 'ctrl_proj_open' in ctrl_path:
                return -1     
            if group == 'JPL1'and 'ctrl_proj_open' in ctrl_path:
                return -1     
        
        # MUN_GISM1 is corrupted, skip
        if group == 'MUN' and model == 'GSM1':
            return -1
        # folder is empty, skip
        elif group == 'IMAU' and exp == 'exp11':
            return -1
        # bed file in NCAR_CISM/expD10 is empty, skip
        elif group == 'NCAR' and exp in ('expD10', 'expD11'):
            return -1
        
        # lookup densities from csv
        subset_densities = densities[(densities.group == group) & (densities.model == model)]
        rhoi = subset_densities.rhoi.values[0]
        rhow = subset_densities.rhow.values[0]
        
        # load data
        if self.ice_sheet == "AIS" and group == 'ULB':
            # ULB uses fETISh for AIS naming, not actual model name (fETISh_16km or fETISh_32km)
            naming_convention = f'{self.ice_sheet}_{group}_fETISh_{exp}.nc'
            
        else:
            naming_convention = f'{self.ice_sheet}_{group}_{model}_{exp}.nc'
        
        # load data
        bed = get_xarray_data(os.path.join(directory, f'topg_{naming_convention}'), ice_sheet=self.ice_sheet)
        thickness = get_xarray_data(os.path.join(directory, f'lithk_{naming_convention}'), ice_sheet=self.ice_sheet )
        mask = get_xarray_data(os.path.join(directory, f'sftgif_{naming_convention}'), ice_sheet=self.ice_sheet )
        ground_mask = get_xarray_data(os.path.join(directory, f'sftgrf_{naming_convention}'), ice_sheet=self.ice_sheet )
        
        # bed = xr.open_dataset(os.path.join(directory, f'topg_{naming_convention}'), decode_times=False)
        # thickness = xr.open_dataset(os.path.join(directory, f'lithk_{naming_convention}'), decode_times=False)
        # mask = xr.open_dataset(os.path.join(directory, f'sftgif_{naming_convention}'), decode_times=False)
        # ground_mask = xr.open_dataset(os.path.join(directory, f'sftgrf_{naming_convention}'), decode_times=False)
        length_time = len(thickness.time)
        # note on decode_times=False -- by doing so, it stays in "days from" rather than trying to infer a type. Makes handling much more predictable.
        
        
        try:
            bed = bed.transpose('x', 'y', 'time', ...)
            thickness = thickness.transpose('x', 'y', 'time', ...)
            mask = mask.transpose('x', 'y', 'time', ...)
            ground_mask = ground_mask.transpose('x', 'y', 'time', ...)
        except ValueError:
            bed = bed.transpose('x', 'y', ...)
            thickness = thickness.transpose('x', 'y', ...)
            mask = mask.transpose('x', 'y', ...)
            ground_mask = ground_mask.transpose('x', 'y', ...)
            
            
        # if time is not a dimension, add copies for each time step
        if 'time' not in bed.dims or bed.dims['time'] == 1:
            try:
                bed = bed.drop_vars(['time',])
            except ValueError:
                pass
            bed = bed.expand_dims(dim={'time': length_time})
            
            if length_time == 86:
                bed['time'] = thickness['time'] # most times just the bed file is missing the time index
            elif length_time > 86:
                if len(thickness.time.values) != len(set(thickness.time.values)): # has duplicates
                    keep_indices = np.unique(thickness['time'], return_index=True)[1] # find non-duplicates
                    bed = bed.isel(time=keep_indices) # only select non-duplicates
                    thickness = thickness.isel(time=keep_indices)
                    mask = mask.isel(time=keep_indices)
                    ground_mask = ground_mask.isel(time=keep_indices)
                else:
                    warnings.warn(f"At least one file in {exp} does not have a time index formatted correctly. Attempting to fix.")
                    start_idx = len(bed.time) - 86
                    bed = bed.sel(time=slice(bed.time.values[start_idx], len(bed.time)))
                    thickness = thickness.sel(time=slice(thickness.time[start_idx], thickness.time[-1]))
                    mask = mask.sel(time=slice(mask.time[start_idx], mask.time[-1]))
                    ground_mask = ground_mask.sel(time=slice(ground_mask.time[start_idx], ground_mask.time[-1]))
                    
                try:
                    bed['time'] = thickness['time'].copy()
                except ValueError:
                    print(f'Cannot fix time index for {exp} due to duplicate index values. Skipped.')
                    return -1
                
            else:
                print(f"Only {len(bed.time)} time points for {exp}. Skipped.")
                return -1
            


        # if -9999 instead of np.nan, replace (come back and optimize? couldn't figure out with xarray)
        if bed.topg[0,0,0] <= -9999. or bed.topg[0,0,0] >= 9999:
            topg = bed.topg.values
            topg[(np.where((topg <= -9999.) | (topg >= 9999)))] = np.nan
            bed['topg'].values = topg
            del topg
            
            lithk = thickness.lithk.values
            lithk[(np.where((lithk <= -9999.) | (lithk >= 9999)))] = np.nan
            thickness['lithk'].values = lithk
            del lithk
            
            sftgif = mask.sftgif.values
            sftgif[(np.where((sftgif <= -9999.) | (sftgif >= 9999)))] = np.nan
            mask['sftgif'].values = sftgif
            del sftgif
            
            sftgrf = ground_mask.sftgrf.values
            sftgrf[(np.where((sftgrf <= -9999.) | (sftgrf >= 9999)))] = np.nan
            ground_mask['sftgrf'].values = sftgrf
            del sftgrf
        
        # converts time (in "days from X" to numpy.datetime64) and subsets time from 2015 to 2100
        
        
        # a few datasets do not have the time index formatted correctly
        if len(bed.time.attrs) == 0:
            
            if len(bed.time) == 86:
                bed['time'] = thickness['time'] # most times just the bed file is missing the time index
            elif len(bed.time) > 86:
                # bed['time'] = thickness['time'].copy()
                warnings.warn(f"At least one file in {exp} does not have a time index formatted correctly. Attempting to fix.")
                start_idx = len(bed.time) - 86
                bed = bed.sel(time=slice(bed.time.values[start_idx], len(bed.time)))
                thickness = thickness.sel(time=slice(thickness.time[start_idx], thickness.time[-1]))
                mask = mask.sel(time=slice(mask.time[start_idx], mask.time[-1]))
                ground_mask = ground_mask.sel(time=slice(ground_mask.time[start_idx], ground_mask.time[-1]))
                
                try:
                    bed['time'] = thickness['time']
                except ValueError:
                    print(f'Cannot fix time index for {exp} due to duplicate index values. Skipped.')
                    return -1
                
            else:
                print(f"Only {len(bed.time)} time points for {exp}. Skipped.")
                return -1
            
        bed = convert_and_subset_times(bed)
        thickness = convert_and_subset_times(thickness)
        mask = convert_and_subset_times(mask)
        ground_mask = convert_and_subset_times(ground_mask)
        length_time = len(thickness.time)

        
        # Interpolate values for x & y, for formatting purposes only, does not get used
        if len(set(thickness.y.values)) != len(scalefac_model):
            bed['x'], bed['y'] = interpolate_values(bed)
            thickness['x'], thickness['y'] = interpolate_values(thickness)
            mask['x'], mask['y'] = interpolate_values(mask)
            ground_mask['x'], ground_mask['y'] = interpolate_values(ground_mask)
        
        # clip masks if they are below 0 or above 1
        if np.min(mask.sftgif.values) < 0 or np.max(mask.sftgif.values) > 1:
            mask['sftgif'] = np.clip(mask.sftgif, 0., 1.)
        if np.min(ground_mask.sftgrf.values) < 0 or np.max(ground_mask.sftgrf.values) > 1:
            ground_mask['sftgrf'] = np.clip(ground_mask.sftgrf, 0., 1.)
        
        # if time is not a dimension, add copies for each time step
        # if 'time' not in bed.dims or bed.dims['time'] == 1:
        #     try:
        #         bed = bed.drop_vars(['time',])
        #     except ValueError:
        #         pass
        #     bed = bed.expand_dims(dim={'time': length_time})
            
        # flip around axes so the order is (x, y, time)
        bed = bed.transpose('x', 'y', 'time', ...)
        bed_data = bed.topg.values
        
        thickness = thickness.transpose('x', 'y', 'time', ...)
        thickness_data = thickness.lithk.values
        
        mask = mask.transpose('x', 'y', 'time', ...)
        mask_data = mask.sftgif.values
        
        ground_mask = ground_mask.transpose('x', 'y', 'time', ...)
        ground_mask_data = ground_mask.sftgrf.values
        
        # for each time step, calculate ivaf
        ivaf = np.zeros(bed_data.shape)
        for i in range(length_time):   
            
            # get data slices for current time         
            thickness_i = thickness_data[:,:,i].copy()
            bed_i = bed_data[:,:,i].copy()
            mask_i = mask_data[:,:,i].copy()
            ground_mask_i = ground_mask_data[:,:,i].copy()
            
            # set data slices to zero where mask = 0 or any value is NaN
            thickness_i[(mask_i == 0) | (np.isnan(mask_i))| (np.isnan(thickness_i))| (np.isnan(ground_mask_i))| (np.isnan(bed_i))] = 0
            bed_i[(mask_i == 0) | (np.isnan(mask_i))| (np.isnan(thickness_i))| (np.isnan(ground_mask_i))| (np.isnan(bed_i))] = 0
            ground_mask_i[(mask_i == 0) | np.isnan(mask_i) | np.isnan(thickness_i)| np.isnan(ground_mask_i)| np.isnan(bed_i)] = 0
            mask_i[(mask_i == 0) | (np.isnan(mask_i))| (np.isnan(thickness_i))| (np.isnan(ground_mask_i))| (np.isnan(bed_i))] = 0
            
            # take min(bed_i, 0)
            bed_i[bed_i > 0] = 0
            
            # calculate IVAF (based on MATLAB processing scripts from Seroussi, 2021)
            hf_i = thickness_i+((rhow/rhoi)*bed_i)   
            masked_output = hf_i * ground_mask_data[:, :, i] * mask_data[:, :, i]
            ivaf[:, :, i] =  masked_output * scalefac_model * (self.resolution*1000)**2
            
            
        
        # subtract out control if for an experment
        ivaf_nc = bed.copy()  # copy file structure and metadata for ivaf file
        if not ctrl_proj:
            #open control dataset
            ivaf_ctrl = xr.open_dataset(ctrl_path, ).transpose('x', 'y', 'time', ...)

            # subtract out control             
            ivaf = ivaf_ctrl.ivaf.values - ivaf
            
        # save ivaf file (copied format from bed_data, change accordingly.)
        ivaf_nc['ivaf'] = (('x', 'y', 'time'), ivaf)
        ivaf_nc = ivaf_nc.drop_vars(['topg',])
        ivaf_nc['sle'] = ivaf_nc.ivaf / 1e9 / 362.5
        ivaf_nc.to_netcdf(os.path.join(directory, f'ivaf_{self.ice_sheet}_{group}_{model}_{exp}.nc'))
        
        

        print(f"{group}_{model}_{exp}: Processing successful.")
        
        return 1

def convert_and_subset_times(dataset,):
    if isinstance(dataset.time.values[0], cftime._cftime.DatetimeNoLeap) or isinstance(dataset.time.values[0], cftime._cftime.Datetime360Day):
        datetimeindex = dataset.indexes['time'].to_datetimeindex()
        dataset['time'] = datetimeindex   

    elif isinstance(dataset.time.values[0], np.float32) or isinstance(dataset.time.values[0], np.float64) or isinstance(dataset.time.values[0], np.int32) or isinstance(dataset.time.values[0], np.int64):
        try:
            units = dataset.time.attrs['units']
        except KeyError:
            units = dataset.time.attrs['unit']
        units = units.replace('days since ', "").split(' ')[0]
        
        if units == '2000-1-0': # VUB AISMPALEO
            units = '2000-1-1'
        elif units == "day": # NCAR CISM exp7 - "day as %Y%m%d.%f"?
            units = '2014-1-1'
            
        if units == 'seconds': # VUW PISM -- seconds since 1-1-1 00:00:00
            start_date = np.datetime64(datetime.strptime('0001-01-01 00:00:00', "%Y-%m-%d %H:%M:%S"))
            dataset['time'] = np.array([start_date + np.timedelta64(int(x), 's') for x in dataset.time.values])
        elif units == '2008-1-1' and dataset.time[-1] == 157785.0: # UAF?
            # every 5 years but still len(time) == 86.. assume we keep them all for 2015-2100
            dataset['time'] = np.array([np.datetime64(datetime.strptime(f'{x}-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")) for x in range(2015, 2101)])
        else:
            try:
                start_date = np.datetime64(datetime.strptime(units.replace("days since ", ""), "%Y-%m-%d"))
            except ValueError:
                start_date = np.datetime64(datetime.strptime(units.replace("days since ", ""), "%d-%m-%Y"))

            dataset['time'] = np.array([start_date + np.timedelta64(int(x), 'D') for x in dataset.time.values])
    else:
        raise ValueError(f"Time values are not recognized: {type(dataset.time.values[0])}")
    
    if len(dataset.time) > 86:
        # make sure the max date is 2100
        # dataset = dataset.sel(time=slice(np.datetime64('2014-01-01'), np.datetime64('2101-01-01')))
        dataset = dataset.sel(time=slice('2012-01-01', '2101-01-01'))
        
        # if you still have more than 86, take the previous 86 values from 2100
        if len(dataset.time) > 86:
            # LSCE GRISLI has two 2015 measurements
            
            # dataset = dataset.sel(time=slice(dataset.time.values[len(dataset.time) - 86], dataset.time.values[-1]))
            start_idx = len(dataset.time) - 86
            dataset = dataset.isel(time=slice(start_idx, len(dataset.time)))
        
    if len(dataset.time) != 86:
        warnings.warn('After subsetting there are still not 86 time points. Go back and check logs.')
        print(f"dataset_length={len(dataset.time)} -- {dataset.attrs}")
            
    return dataset

def get_model_densities(zenodo_directory: str, output_path: str=None):
    """
    Extracts values for rhoi and rhow from NetCDF files in the specified directory and returns a pandas DataFrame
    containing the group, model, rhoi, and rhow values for each file.

    Args:
        zenodo_directory (str): The path to the directory containing the NetCDF files.
        output_path (str, optional): The path to save the resulting DataFrame as a CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the group, model, rhoi, and rhow values for each file.
    """
    results = []
    for root, dirs, files in os.walk(zenodo_directory):
        for file in files:
            if file.endswith(".nc"):  # Check if the file is a NetCDF file
                file_path = os.path.join(root, file)
                try:
                    # Open the NetCDF file using xarray
                    dataset = xr.open_dataset(file_path, decode_times=False).transpose('x', 'y', 'time', ...)

                    # Extract values for rhoi and rhow
                    if 'rhoi' in dataset and 'rhow' in dataset:
                        rhoi_values = dataset['rhoi'].values
                        rhow_values = dataset['rhow'].values

                        # Append the filename and values to the results list
                        results.append({
                            'filename': file,
                            'rhoi': rhoi_values,
                            'rhow': rhow_values
                        })

                    # Close the dataset
                    dataset.close()
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    densities = []      
    for file in results:
        if 'ctrl_proj' in file['filename'] or 'hist' in file['filename']:
            continue
        
        elif 'ILTS' in file['filename']:
            fp = file['filename'].split('_')
            group = 'ILTS_PIK'
            model = fp[-2]  
            
        elif 'ULB_fETISh' in file['filename']:
            fp = file['filename'].split('_')
            group = 'ULB'
            model = "fETISh_32km" if '32km' in file['filename'] else "fETISh_16km"
            
        else:
            fp = file['filename'].split('_')
            group = fp[-3]
            model = fp[-2]
        densities.append([group, model, file['rhoi'], file['rhow']])

    df = pd.DataFrame(densities, columns=['group', 'model', 'rhoi', 'rhow'])
    df['rhoi'], df['rhow'] = df.rhoi.astype('float'), df.rhow.astype('float')
    df = df.drop_duplicates()
    
    ice_sheet = "AIS" if "AIS" in file['filename'] else "GIS"
    
    if output_path is not None:
        if output_path.endswith('/'):
            df.to_csv(f"{output_path}/{ice_sheet}_densities.csv", index=False)
        else:
            df.to_csv(output_path, index=False)
    
    return df

def interpolate_values(data):
    """
    Interpolates missing values in the x and y dimensions of the input NetCDF data using linear interpolation.
    
    Args:
        data: A NetCDF file containing x and y dimensions with missing values.
        
    Returns:
        A tuple containing the interpolated x and y arrays.
    """
    y = pd.Series(data.y.values)
    y = y.replace(0, np.NaN)
    y = np.array(y.interpolate())
    
    # first and last are NaNs, replace with correct values
    y[0] = y[1] - (y[2]-y[1])
    y[-1] = y[-2] + (y[-2]-y[-3])
    
    x = pd.Series(data.x.values)
    x = x.replace(0, np.NaN)
    x = np.array(x.interpolate())
    
    # first and last are NaNs, replace with correct values
    x[0] = x[1] - (x[2]-x[1])
    x[-1] = x[-2] + (x[-2]-x[-3])
    
    return x, y


class DimensionalityReducer:
    def __init__(self, forcing_dir, projection_dir, output_dir, ice_sheet=None):
        super().__init__()
        if forcing_dir is None:
            raise ValueError("Forcing directory must be specified.")
        if output_dir is None:
            raise ValueError("Output directory must be specified.")
        self.forcing_dir = forcing_dir
        self.projection_dir = projection_dir
        self.output_dir = output_dir
        self.forcing_paths = {'all': None, 'atmosphere': None, 'ocean': None}
        
        # check inputs
        if os.path.exists(f"{self.output_dir}/pca_models/"):
            self.pca_model_directory = f"{self.output_dir}/pca_models/"
        else:
            self.pca_model_directory = None
            
        if os.path.exists(f"{self.output_dir}/scalers/"):
            self.scaler_directory = f"{self.output_dir}/scalers/"
        else:
            self.scaler_directory = None
            
        if ice_sheet not in ('AIS', 'GrIS'):
            raise ValueError("Ice sheet must be specified and must be 'AIS' or 'GrIS'.")
        else:
            self.ice_sheet = ice_sheet
            
        
        if self.ice_sheet.lower() == 'gris':
            atmospheric_files = get_all_filepaths(path=self.forcing_dir, filetype='nc', contains='Atmosphere_Forcing/aSMB_observed/v1', )
            atmospheric_files = [x for x in atmospheric_files if 'combined' in x]
            
            # files in atmopheric directory are separated by year, needs to be combined
            if not atmospheric_files:
                combine_gris_forcings(self.forcing_dir)
                
            oceanic_files = get_all_filepaths(path=self.forcing_dir, filetype='nc', contains='Ocean_Forcing/Melt_Implementation/v4', )
            self.forcing_paths['all'] = atmospheric_files + oceanic_files
            self.forcing_paths['atmosphere'] = atmospheric_files
            self.forcing_paths['ocean'] = oceanic_files
        else:            
            all_forcing_fps = get_all_filepaths(path=self.forcing_dir, filetype='nc', contains='1995-2100', not_contains='Ice_Shelf_Fracture')
            self.forcing_paths['all'] = [x for x in all_forcing_fps if '8km' in x and 'v1' not in x]
            self.forcing_paths['atmosphere'] = [x for x in self.forcing_paths['all'] if "Atmosphere_Forcing" in x]
            self.forcing_paths['ocean'] = [x for x in self.forcing_paths['all'] if "Ocean_Forcing" in x]
            
        
        
        all_projection_fps = get_all_filepaths(path=self.projection_dir, filetype='nc', contains='ivaf', not_contains='ctrl_proj')
        self.projection_paths = all_projection_fps
        

                
        
    # def reduce_dimensionlity(self, forcing_dir: str=None, output_dir: str=None):
            # generate pca models
            # convert each forcing file to pca space
        
        
    def generate_pca_models(self, ):
        """
        Generate principal component analysis (PCA) models for atmosphere and ocean variables. 
        
        Parameters:
        - atmosphere_fps (list): List of file paths for atmosphere data.
        - ocean_fps (list): List of file paths for ocean data.
        - save_dir (str): Directory to save the generated PCA models and results.
        
        Returns:
            int: 0 if successful.
        """
        
        # check inputs
        if not os.path.exists(f"{self.output_dir}/pca_models/"):
            os.mkdir(f"{self.output_dir}/pca_models/")
        self.pca_model_directory = f"{self.output_dir}/pca_models/"
        if not os.path.exists(f"{self.output_dir}/scalers/"):
            os.mkdir(f"{self.output_dir}/scalers/")
        self.scaler_directory = f"{self.output_dir}/scalers/"
        
        # Train PCA models for each atmospheric and oceanic forcing variable and save
        if self.ice_sheet == 'AIS':
            self._generate_ais_atmosphere_pcas(self.forcing_paths['atmosphere'], self.pca_model_directory, scaler_dir=self.scaler_directory)
            self._generate_ais_ocean_pcas(self.forcing_paths['ocean'], self.pca_model_directory, scaler_dir=self.scaler_directory)
        else:
            self._generate_gris_atmosphere_pcas(self.forcing_paths['atmosphere'], self.pca_model_directory, scaler_dir=self.scaler_directory)
            self._generate_gris_ocean_pcas(self.forcing_paths['ocean'], self.pca_model_directory, scaler_dir=self.scaler_directory)
        
        # Train PCA model for SLE and save
        sle_paths = get_all_filepaths(path=self.projection_dir, filetype='nc', contains='ivaf', not_contains='ctrl')
        self._generate_sle_pca(sle_paths, save_dir=self.pca_model_directory, scaler_dir=self.scaler_directory)
        
        return 0
    
    def convert_forcings(self, num_pcs='99%', forcing_files: list=None, pca_model_directory: str=None, output_dir: str=None):
        """
        Converts atmospheric and oceanic forcing files to PCA space using pretrained PCA models.
        
        Args:
            forcing_files (list, optional): List of specific forcing files to convert. If not provided, all files in the directory will be used. Default is None.
            pca_model_directory (str, optional): Directory containing the pretrained PCA models. If not provided, the directory specified during object initialization will be used. Default is None.
            output_dir (str, optional): Directory to save the converted files. If not provided, the directory specified during object initialization will be used. Default is None.
        
        Returns:
            int: 0 indicating successful conversion.
        """
        
        # check inputs for validity
        output_dir = self.output_dir if output_dir is None else output_dir
        if self.pca_model_directory is None and pca_model_directory is None:
            raise ValueError("PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first.")
        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory
        
        # if user supplies specific forcing files (rather than entire directory), use that instead
        # TODO: test this..
        if forcing_files is not None:
            warnings.warn("By using specific forcing files, forcing_paths attribute will be overwritten.")
            self.forcing_paths['all'] = forcing_files
            self.forcing_paths['atmosphere'] = [x for x in self.forcing_paths['all'] if "Atmosphere_Forcing" in x]
            self.forcing_paths['ocean'] = [x for x in self.forcing_paths['all'] if "Ocean_Forcing" in x]
            
        
        # ATMOSPHERIC FORCINGS
        
        if not os.path.exists(f"{output_dir}/forcings/"):
            os.mkdir(f"{output_dir}/forcings/")
            
        
        # for each atmospheric forcing file, convert each variable to PCA space with pretrained PCA model
        atmospheric_paths_loop = tqdm(enumerate(self.forcing_paths['atmosphere']), total=len(self.forcing_paths['atmosphere']))
        for i, path in atmospheric_paths_loop:
            atmospheric_paths_loop.set_description(f"Converting atmospheric file #{i+1}/{len(self.forcing_paths['atmosphere'])}")
            
            dataset = xr.open_dataset(path, decode_times=False, engine='netcdf4').transpose('time', 'y', 'x', ...)  # open the dataset
            if len(dataset.dims) > 3:
                drop_dims = [x for x in list(dataset.dims) if x not in ('time', 'x', 'y')]
                dataset = dataset.drop_dims(drop_dims)
            # dataset = convert_and_subset_times(dataset)
            forcing_name = path.replace('.nc', '').split('/')[-1]  # get metadata (model, ssp, etc.)
            
            # transform each variable in the dataset with their respective trained PCA model
            transformed_data = {}
            for var in ['evspsbl_anomaly', 'mrro_anomaly', 'pr_anomaly', 'smb_anomaly', 'ts_anomaly']:
                try:
                    transformed = self.transform(dataset[var].values, var_name=var, num_pcs=num_pcs, pca_model_directory=self.pca_model_directory, scaler_directory=self.scaler_directory)
                except KeyError: # if a variable is missing (usually mrro_anomaly), skip it
                    continue
                transformed_data[var] = transformed  # store in dict with structure {'var_name': transformed_var}
            
            # create a dataframe with rows corresponding to time (106 total) and columns corresponding to each variables principal components
            compiled_transformed_forcings = pd.DataFrame()
            for var in transformed_data.keys():
                var_df = pd.DataFrame(transformed_data[var], columns=[f"{var}_pc{i+1}" for i in range(transformed_data[var].shape[1])])
                compiled_transformed_forcings = pd.DataFrame(pd.concat([compiled_transformed_forcings, var_df], axis=1))
            

            pd.DataFrame(compiled_transformed_forcings).to_csv(f"{output_dir}/forcings/PCA_{forcing_name}.csv", index=False)
        
        print(f"{len(self.forcing_paths['atmosphere'])}/{len(self.forcing_paths['atmosphere'])} atmospheric forcing files converted to PCA space.")
        print(f'Finished converting atmospheric forcings to PCA space, files outputted to {output_dir}.')
        
        
        # OCEANIC FORCINGS
        
        # for each ocean forcing file, convert each variable to PCA space with pretrained PCA model
        ocean_fps_loop = tqdm(enumerate(self.forcing_paths['ocean']), total=len(self.forcing_paths['ocean']))
        for i, path in ocean_fps_loop:
            ocean_fps_loop.set_description(f"Converting oceanic file #{i+1}/{len(self.forcing_paths['ocean'])}")
            
            # open the dataset
            forcing_name = path.replace('.nc', '').split('/')[-1]  # get metadata (model, ssp, etc.)
            
            # get variable name by splitting the filepath name
            if self.ice_sheet == 'AIS':
                var = self.forcing_paths['ocean'][i].split('/')[-1].split('_')[-4]
            else:
                metadata = self.forcing_paths['ocean'][i].split('/')[-1].split('_')
                if 'basinRunoff' in metadata:
                    var = 'basin_runoff'
                elif 'oceanThermalForcing' in metadata:
                    var = 'thermal_forcing'
                else:
                    var = self.forcing_paths['ocean'][i].split('/')[-1].split('_')[-2]
            if var == 'forcing' or var == 'thermal':
                var = 'thermal_forcing'
            
            # get forcing array (requires mean value over z dimensions, see get_xarray_data())
            forcing_array = get_xarray_data(path, var_name=var, ice_sheet=self.ice_sheet)
            # forcing_array = np.nan_to_num(forcing_array)  # deal with np.nans
            # forcing_array = forcing_array[:, ~(np.isnan(forcing_array).any(axis=0))]
            
            # transform each variable in the dataset with their respective trained PCA model
            transformed_data = {}
            transformed = self.transform(forcing_array, var_name=var, num_pcs=num_pcs, pca_model_directory=self.pca_model_directory, scaler_directory=self.scaler_directory)
            transformed_data[var] = transformed  # store in dict with structure {'var_name': transformed_var}
            
            # create a dataframe with rows corresponding to time (86 total) and columns corresponding to each variables principal components
            variable_df = pd.DataFrame(transformed_data[var], columns=[f"{var}_pc{i+1}" for i in range(transformed_data[var].shape[1])])
            variable_df.to_csv(f"{output_dir}/forcings/PCA_{forcing_name}.csv", index=False)
            
            
        print(f"{len(self.forcing_paths['ocean'])}/{len(self.forcing_paths['ocean'])} oceanic forcing files converted to PCA space.")
        print(f'Finished converting oceanic forcings to PCA space, files outputted to {output_dir}.')
        
                
        return 0
        
        
    def convert_projections(self, num_pcs='99.99%', projection_files: list=None, pca_model_directory: str=None, output_dir: str=None):
        
        # check inputs for validity
        output_dir = self.output_dir if output_dir is None else output_dir
        if self.pca_model_directory is None and pca_model_directory is None:
            raise ValueError("PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first.")
        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory
        
        # if user supplies specific projection files (rather than entire directory), use that instead
        if projection_files is not None:
            warnings.warn("By using specific projection files, projection_paths attribute will be overwritten.")
            self.projection_paths = projection_files
        
        # make a folder in output directory for converted projections
        if not os.path.exists(f"{output_dir}/projections/"):
            os.mkdir(f"{output_dir}/projections/")
            
        # for each projection file, convert ivaf to PCA space with pretrained PCA model
        projection_paths_loop = tqdm(enumerate(self.projection_paths), total=len(self.projection_paths))
        for i, path in projection_paths_loop:
            projection_paths_loop.set_description(f"Converting projection file #{i+1}/{len(self.projection_paths)}")
            
            # get forcing array (requires mean value over z dimensions, see get_xarray_data())
            try:
                projection_array = get_xarray_data(path, var_name='sle', ice_sheet=self.ice_sheet)
            except:
                projection_array = get_xarray_data(path, var_name='ivaf', ice_sheet=self.ice_sheet)
                projection_array = projection_array / 1e9 / 362.5
                
            
            # nan_indices = np.argwhere(np.isnan(projection_array))
            # print(len(nan_indices))
            # continue
            
            # projection_array = np.nan_to_num(projection_array)  # deal with np.nans
            var = 'sle'
            # projection_array = np.nan_to_num(projection_array)  # there shouldn't be nans...
            projection_name = path.replace('.nc', '').split('/')[-1]  # get metadata (model, ssp, etc.)
            
            # transform each variable in the dataset with their respective trained PCA model
            transformed_data = {}
            transformed = self.transform(projection_array, var_name=var, num_pcs=num_pcs, pca_model_directory=self.pca_model_directory, scaler_directory=self.scaler_directory)
            transformed_data[var] = transformed  # store in dict with structure {'var_name': transformed_var}
            
            # create a dataframe with rows corresponding to time (86 total) and columns corresponding to each variables principal components
            variable_df = pd.DataFrame(transformed_data[var], columns=[f"{var}_pc{i+1}" for i in range(transformed_data[var].shape[1])])
            variable_df.to_csv(f"{output_dir}/projections/PCA_{projection_name}.csv", index=False)
        
        print(f"{len(self.projection_paths)}/{len(self.projection_paths)} projection files converted to PCA space.")
        print(f'Finished converting projections to PCA space, files outputted to {output_dir}.')
    
    

    def _generate_ais_atmosphere_pcas(self, atmosphere_fps: list, save_dir: str, scaler_dir: str=None):
        """
        Generate principal component analysis (PCA) for atmospheric variables.

        Args:
            atmosphere_fps (list): List of file paths to atmospheric CMIP files.
            save_dir (str): Directory to save the PCA results.

        Returns:
            int: 0 if successful.
        """
        
        # if no separate directory for saving scalers is specified, use the pca save_dir
        if scaler_dir is None:
            scaler_dir = save_dir
        
        # for each variable
        
        var_names = ['pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly', 'smb_anomaly', 'ts_anomaly']
        var_names_loop = tqdm(enumerate(var_names), total=len(var_names))
        for i, var in var_names_loop:
            var_names_loop.set_description(f"Processing atmospheric PCA #{i+1}/{len(var_names)}")
            variable_array = np.zeros([len(atmosphere_fps), 86, 761*761])
            
            # loop through each atmospheric CMIP file and combine them into one big array
            for i, fp in enumerate(atmosphere_fps):
                
                # get the variable you need (rather than the entire dataset)
                dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet)
                data_array = convert_and_subset_times(dataset)
                try:
                    data_flattened = data_array[var].values.reshape(86, 761*761)
                except KeyError:
                    data_flattened = np.nan
                # store it in the total array
                variable_array[i, :, :] = data_flattened     
                

            # deal with np.nans -- since it's an anomaly, replace with 0
            variable_array = np.nan_to_num(variable_array)   
            
            # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
            variable_array = variable_array.reshape(len(atmosphere_fps)*86, 761*761)
            
            # scale data
            variable_scaler = StandardScaler()
            variable_scaler.fit(variable_array)
            variable_array = variable_scaler.transform(variable_array)
            
            # run PCA
            pca, _ = self._run_PCA(variable_array, num_pcs=1000)
            
            # get percent explained
            exp_var_pca = pca.explained_variance_ratio_
            cum_sum_eigenvalues = np.cumsum(exp_var_pca)
            arg_90 = np.argmax(cum_sum_eigenvalues>0.90)+1
            print(f"Variable {var} has {arg_90} PCs that explain 90% of the variance")
            
            # output pca object
            save_path = f"{save_dir}/AIS_pca_{var}_{arg_90}pcs.pkl"
            pkl.dump(pca, open(save_path,"wb"))
            # and scaler
            save_path = f"{scaler_dir}/AIS_{var}_scaler.pkl"
            pkl.dump(variable_scaler, open(save_path,"wb"))
            
        return 0
    
    def _generate_ais_ocean_pcas(self, ocean_fps: list, save_dir: str, scaler_dir: str=None):
        """
        Generate principal component analysis (PCA) for ocean variables.

        Args:
            ocean_fps (list): List of file paths for ocean variables.
            save_dir (str): Directory to save the PCA results.

        Returns:
            int: 0 if PCA generation is successful, -1 otherwise.
        """
        
        if scaler_dir is None:
            scaler_dir = save_dir
        
        thermal_forcing_fps = [x for x in ocean_fps if 'thermal_forcing' in x]
        salinity_fps = [x for x in ocean_fps if 'salinity' in x]
        temperature_fps = [x for x in ocean_fps if 'temperature' in x]
        
        thermal_forcing_array = np.zeros([len(thermal_forcing_fps), 86, 761*761])
        salinity_array = np.zeros([len(salinity_fps), 86, 761*761])
        temperature_array = np.zeros([len(temperature_fps), 86, 761*761])
        
        # get the variables you need (rather than the entire dataset)
        print('Processing thermal_forcing PCA model.')
        for i, fp in enumerate(thermal_forcing_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet)
            data_array = convert_and_subset_times(dataset)
            data_flattened = data_array['thermal_forcing'].values.reshape(86, 761*761)
            thermal_forcing_array[i, :, :] = data_flattened # store
        print('Processing salinity PCA model.')
        for i, fp in enumerate(salinity_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet)
            data_array = convert_and_subset_times(dataset)
            data_flattened = data_array['salinity'].values.reshape(86, 761*761)
            salinity_array[i, :, :] = data_flattened # store
        print('Processing temperature PCA model.')
        for i, fp in enumerate(temperature_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet)
            data_array = convert_and_subset_times(dataset)
            data_flattened = data_array['temperature'].values.reshape(86, 761*761)
            temperature_array[i, :, :] = data_flattened
            
        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        thermal_forcing_array = thermal_forcing_array.reshape(len(thermal_forcing_fps)*86, 761*761)
        salinity_array = salinity_array.reshape(len(salinity_fps)*86, 761*761)
        temperature_array = temperature_array.reshape(len(temperature_fps)*86, 761*761)
        
        # remove nans
        thermal_forcing_array = np.nan_to_num(thermal_forcing_array)
        salinity_array = np.nan_to_num(salinity_array)
        temperature_array = np.nan_to_num(temperature_array)
        
        # scale data
        therm_scaler = StandardScaler()
        therm_scaler.fit(thermal_forcing_array)
        thermal_forcing_array = therm_scaler.transform(thermal_forcing_array)
        
        salinity_scaler = StandardScaler()
        salinity_scaler.fit(salinity_array)
        salinity_array = salinity_scaler.transform(salinity_array)
        
        temp_scaler = StandardScaler()
        temp_scaler.fit(temperature_array)
        temperature_array = temp_scaler.transform(temperature_array)
        
        # run PCA
        pca_tf, _ = self._run_PCA(thermal_forcing_array, num_pcs=1000)
        pca_sal, _ = self._run_PCA(salinity_array, num_pcs=1000)
        pca_temp, _ = self._run_PCA(temperature_array, num_pcs=1000)
        
        # get percent explained
        tf_exp_var_pca = pca_tf.explained_variance_ratio_
        tf_cum_sum_eigenvalues = np.cumsum(tf_exp_var_pca)
        tf_arg_90 = np.argmax(tf_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/AIS_pca_thermal_forcing_{tf_arg_90}pcs.pkl"
        pkl.dump(pca_tf, open(save_path,"wb"))

        
        sal_exp_var_pca = pca_sal.explained_variance_ratio_
        sal_cum_sum_eigenvalues = np.cumsum(sal_exp_var_pca)
        sal_arg_90 = np.argmax(sal_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/AIS_pca_salinity_{sal_arg_90}pcs.pkl"
        pkl.dump(pca_sal, open(save_path,"wb"))
        
        temp_exp_var_pca = pca_temp.explained_variance_ratio_
        temp_cum_sum_eigenvalues = np.cumsum(temp_exp_var_pca)
        temp_arg_90 = np.argmax(temp_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/AIS_pca_temperature_{temp_arg_90}pcs.pkl"
        pkl.dump(pca_temp, open(save_path,"wb"))
        
        # save scalers 
        save_path = f"{scaler_dir}/AIS_thermal_forcing_scaler.pkl"
        pkl.dump(therm_scaler, open(save_path,"wb"))
        
        save_path = f"{scaler_dir}/AIS_temperature_scaler.pkl"
        pkl.dump(temp_scaler, open(save_path,"wb"))
        
        save_path = f"{scaler_dir}/AIS_salinity_scaler.pkl"
        pkl.dump(salinity_scaler, open(save_path,"wb"))
        
        return 0
    
    def _generate_gris_atmosphere_pcas(self, atmosphere_fps: list, save_dir: str, scaler_dir: str=None):
        
        # if no separate directory for saving scalers is specified, use the pca save_dir
        if scaler_dir is None:
            scaler_dir = save_dir
            
        aSMB_fps = [x for x in atmosphere_fps if 'aSMB-combined' in x]
        aST_fps = [x for x in atmosphere_fps if 'aST-combined' in x]
        
        # for each variable
        flattened_xy_dim = 337*577
        smb_forcing_array = np.zeros([len(aSMB_fps), 86, flattened_xy_dim])
        st_forcing_array = np.zeros([len(aST_fps), 86, flattened_xy_dim])
        
        # get the variables you need (rather than the entire dataset)
        print('Processing aSMB PCA model.')
        for i, fp in enumerate(aSMB_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet)
            data_array = convert_and_subset_times(dataset)
            data_flattened = data_array['aSMB'].values.reshape(86, flattened_xy_dim)
            smb_forcing_array[i, :, :] = data_flattened # store
        print('Processing aST PCA model.')
        for i, fp in enumerate(aST_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet)
            data_array = convert_and_subset_times(dataset)
            data_flattened = data_array['aST'].values.reshape(86, flattened_xy_dim)
            st_forcing_array[i, :, :] = data_flattened # store
            
        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        smb_forcing_array = smb_forcing_array.reshape(len(aSMB_fps)*len(data_array.time), flattened_xy_dim)
        st_forcing_array = st_forcing_array.reshape(len(aST_fps)*len(data_array.time), flattened_xy_dim)
        
        # remove nans
        smb_forcing_array = np.nan_to_num(smb_forcing_array)
        st_forcing_array = np.nan_to_num(st_forcing_array)
        
        # scale data
        smb_scaler = StandardScaler()
        smb_scaler.fit(smb_forcing_array)
        smb_forcing_array = smb_scaler.transform(smb_forcing_array)
        
        st_scaler = StandardScaler()
        st_scaler.fit(st_forcing_array)
        st_forcing_array = st_scaler.transform(st_forcing_array)
        
        # run PCA
        pca_smb, _ = self._run_PCA(smb_forcing_array, num_pcs=1000)
        pca_st, _ = self._run_PCA(st_forcing_array, num_pcs=1000)
        
        # get percent explained
        smb_exp_var_pca = pca_smb.explained_variance_ratio_
        smb_cum_sum_eigenvalues = np.cumsum(smb_exp_var_pca)
        smb_arg_90 = np.argmax(smb_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/GrIS_pca_aSMB_{smb_arg_90}pcs.pkl"
        pkl.dump(pca_smb, open(save_path,"wb"))

        
        st_exp_var_pca = pca_st.explained_variance_ratio_
        st_cum_sum_eigenvalues = np.cumsum(st_exp_var_pca)
        st_arg_90 = np.argmax(st_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/GrIS_pca_aST_{st_arg_90}pcs.pkl"
        pkl.dump(pca_st, open(save_path,"wb"))
        
        
        # save scalers 
        save_path = f"{scaler_dir}/GrIS_aSMB_scaler.pkl"
        pkl.dump(smb_scaler, open(save_path,"wb"))
        
        save_path = f"{scaler_dir}/GrIS_aST_scaler.pkl"
        pkl.dump(st_scaler, open(save_path,"wb"))
        
            
        return 0
    
    def _generate_gris_ocean_pcas(self, ocean_fps: list, save_dir: str, scaler_dir: str=None):
        
        # if no separate directory for saving scalers is specified, use the pca save_dir
        if scaler_dir is None:
            scaler_dir = save_dir
            
        basin_runoff_fps = [x for x in ocean_fps if 'basinRunoff' in x]
        thermal_forcing_fps = [x for x in ocean_fps if 'oceanThermalForcing' in x]
        
        # for each variable
        flattened_xy_dim = 337*577
        basin_runoff_array = np.zeros([len(basin_runoff_fps), 86, flattened_xy_dim])
        thermal_forcing_array = np.zeros([len(thermal_forcing_fps), 86, flattened_xy_dim])
        
        # get the variables you need (rather than the entire dataset)
        print('Processing basin_runoff PCA model.')
        for i, fp in enumerate(basin_runoff_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet)
            data_array = convert_and_subset_times(dataset)
            data_flattened = data_array['basin_runoff'].values.reshape(86, flattened_xy_dim)
            basin_runoff_array[i, :, :] = data_flattened # store
        print('Processing thermal_forcing PCA model.')
        for i, fp in enumerate(thermal_forcing_fps):
            dataset = get_xarray_data(fp, ice_sheet=self.ice_sheet)
            data_array = convert_and_subset_times(dataset)
            data_flattened = data_array['thermal_forcing'].values.reshape(86, flattened_xy_dim)
            thermal_forcing_array[i, :, :] = data_flattened # store
            
        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        basin_runoff_array = basin_runoff_array.reshape(len(basin_runoff_fps)*len(data_array.time), flattened_xy_dim)
        thermal_forcing_array = thermal_forcing_array.reshape(len(thermal_forcing_fps)*len(data_array.time), flattened_xy_dim)
        
        # remove nans
        basin_runoff_array = np.nan_to_num(basin_runoff_array)
        thermal_forcing_array = np.nan_to_num(thermal_forcing_array)
        
        # scale data
        basin_runoff_scaler = StandardScaler()
        basin_runoff_scaler.fit(basin_runoff_array)
        basin_runoff_array = basin_runoff_scaler.transform(basin_runoff_array)
        
        thermal_forcing_scaler = StandardScaler()
        thermal_forcing_scaler.fit(thermal_forcing_array)
        thermal_forcing_array = thermal_forcing_scaler.transform(thermal_forcing_array)
        
        # run PCA
        pca_br, _ = self._run_PCA(basin_runoff_array, num_pcs=1000)
        pca_tf, _ = self._run_PCA(thermal_forcing_array, num_pcs=1000)
        
        # get percent explained
        br_exp_var_pca = pca_br.explained_variance_ratio_
        br_cum_sum_eigenvalues = np.cumsum(br_exp_var_pca)
        br_arg_90 = np.argmax(br_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/GrIS_pca_basin_runoff_{br_arg_90}pcs.pkl"
        pkl.dump(pca_br, open(save_path,"wb"))

        
        tf_exp_var_pca = pca_tf.explained_variance_ratio_
        tf_cum_sum_eigenvalues = np.cumsum(tf_exp_var_pca)
        tf_arg_90 = np.argmax(tf_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/GrIS_pca_thermal_forcing_{tf_arg_90}pcs.pkl"
        pkl.dump(pca_tf, open(save_path,"wb"))
        
        
        # save scalers 
        save_path = f"{scaler_dir}/GrIS_basin_runoff_scaler.pkl"
        pkl.dump(basin_runoff_scaler, open(save_path,"wb"))
        
        save_path = f"{scaler_dir}/GrIS_thermal_forcing_scaler.pkl"
        pkl.dump(thermal_forcing_scaler, open(save_path,"wb"))
        
            
        return 0
    
    def _generate_sle_pca(self, sle_fps: list, save_dir: str, scaler_dir=None):
        """
        Generate principal component analysis (PCA) for sea level equivalent (SLE) variables.

        Args:
            sle_fps (list): List of file paths for SLE variables.
            save_dir (str): Directory to save the PCA results.

        Returns:
            int: 0 if PCA generation is successful, -1 otherwise.
        """
        
        if scaler_dir is None:
            scaler_dir = save_dir
            
        if self.ice_sheet == 'AIS':
            flattened_xy_dim = 761*761
        else:
            flattened_xy_dim = 337*577
        
        sle_array = np.zeros([len(sle_fps), 86, flattened_xy_dim])
        # sle_array = np.zeros([4, 86, 761*761])
        # sle_fps = sle_fps[0:4]
        # loop through each SLE (IVAF) projection file
        sle_fps_loop = tqdm(enumerate(sle_fps), total=len(sle_fps))
        for i, fp in sle_fps_loop:
            sle_fps_loop.set_description(f"Aggregating SLE file #{i+1}/{len(sle_fps)}")
            # get the variable
            try:
                data_flattened = get_xarray_data(fp, var_name="sle", ice_sheet=self.ice_sheet)
            except:
                data_flattened = get_xarray_data(fp, var_name="ivaf", ice_sheet=self.ice_sheet)
                data_flattened = data_flattened / 1e9 / 362.5
            
            # store it in the total array
            sle_array[i, :, :] = data_flattened 
                
        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        sle_array = sle_array.reshape(len(sle_fps)*86, flattened_xy_dim)
        
        # since the array is so large (350*85, 761*761) = (29750, 579121), randomly sample N rows and run PCA
        sle_array = sle_array[np.random.choice(sle_array.shape[0], 1590, replace=False), :]
        
        # deal with np.nans (ask about later)
        # nan_mask = np.array(pd.read_csv(r'/users/pvankatw/research/current/nan_mask.csv'))
        # non_nan_args = np.argwhere(nan_mask == 0).squeeze()
        # sle_array = sle_array[:, non_nan_args]
        sle_array = np.nan_to_num(sle_array) 
        
        # scale sle
        scaler = StandardScaler()
        scaler.fit(sle_array)
        sle_array = scaler.transform(sle_array)
        
        # run pca
        pca, _ = self._run_PCA(sle_array, num_pcs=1000,)
        
        # get percent explained
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        arg_90 = np.argmax(cum_sum_eigenvalues>0.90)+1
        print(f"Variable SLE has {arg_90} PCs that explain 90% of the variance")
        
        # output pca object
        save_path = f"{save_dir}/{self.ice_sheet}_pca_sle_{arg_90}pcs.pkl"
        pkl.dump(pca, open(save_path,"wb"))
        
        # and scaler
        save_path = f"{scaler_dir}/{self.ice_sheet}_sle_scaler.pkl"
        pkl.dump(scaler, open(save_path,"wb"))
        
        
        return 0
    
    def _run_PCA(self, variable_array, num_pcs=None, randomized=False):
        """
        Runs Principal Component Analysis (PCA) on the given variable array.

        Args:
            variable_array (array-like): The input array containing the variables.
            num_pcs (int, optional): The number of principal components to keep. 
                If not specified, all components will be kept.

        Returns:
            tuple: A tuple containing the fitted PCA model and the transformed array.

        """
        solver = 'randomized' if randomized else 'auto'
        if not num_pcs:
            pca = PCA(svd_solver=solver)
        else:
            pca = PCA(n_components=num_pcs, svd_solver=solver)

        pca = pca.fit(variable_array)
        pca_array = pca.transform(variable_array)
        return pca, pca_array
    
    def _load_pca_models(self, pca_model_directory, var_name='all'):
        if self.pca_model_directory is None and pca_model_directory is None:
            raise ValueError("PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first.")
        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory
        
            
        pca_models_paths = os.listdir(self.pca_model_directory)
        pca_models_paths = [x for x in pca_models_paths if 'pca' in x and self.ice_sheet in x]
        
        if self.ice_sheet == 'AIS':
            
            if var_name not in ['all', 'evspsbl_anomaly', 'mrro_anomaly', 'pr_anomaly', 'smb_anomaly', 'ts_anomaly', 'thermal_forcing', 'salinity', 'temperature', 'sle', None]:
                raise ValueError(f"Variable name {var_name} not recognized.")
            
            if var_name == 'all' or var_name is None:
                evspsbl_model = [x for x in pca_models_paths if 'evspsbl' in x][0]
                mrro_model = [x for x in pca_models_paths if 'mrro' in x][0]
                pr_model = [x for x in pca_models_paths if 'pr' in x][0]
                smb_model = [x for x in pca_models_paths if 'smb' in x][0]
                ts_model = [x for x in pca_models_paths if 'ts' in x][0]
                thermal_forcing_model = [x for x in pca_models_paths if 'thermal_forcing' in x][0]
                salinity_model = [x for x in pca_models_paths if 'salinity' in x][0]
                temperature_model = [x for x in pca_models_paths if 'temperature' in x][0]
                sle_model = [x for x in pca_models_paths if 'sle' in x][0]
                    
                pca_models = dict(
                    evspsbl_anomaly=pkl.load(open(f"{self.pca_model_directory}/{evspsbl_model}", "rb")),
                    mrro_anomaly=pkl.load(open(f"{self.pca_model_directory}/{mrro_model}", "rb")),
                    pr_anomaly=pkl.load(open(f"{self.pca_model_directory}/{pr_model}", "rb")),
                    smb_anomaly=pkl.load(open(f"{self.pca_model_directory}/{smb_model}", "rb")),
                    ts_anomaly=pkl.load(open(f"{self.pca_model_directory}/{ts_model}", "rb")),
                    thermal_forcing=pkl.load(open(f"{self.pca_model_directory}/{thermal_forcing_model}", "rb")),
                    salinity=pkl.load(open(f"{self.pca_model_directory}/{salinity_model}", "rb")),
                    temperature=pkl.load(open(f"{self.pca_model_directory}/{temperature_model}", "rb")),
                    sle=pkl.load(open(f"{self.pca_model_directory}/{sle_model}", "rb")),
                )
            else:
                pca_models = {}
                model_path = [x for x in pca_models_paths if var_name in x][0]
                pca_models[var_name] = pkl.load(open(f"{self.pca_model_directory}/{model_path}", "rb"))
        else:
            if var_name not in ['all', 'aST', 'aSMB', 'basin_runoff', 'thermal_forcing','sle', None]:
                raise ValueError(f"Variable name {var_name} not recognized.")
            
            if var_name == 'all' or var_name is None:
                aSMB_model = [x for x in pca_models_paths if 'aSMB' in x][0]
                aST_model = [x for x in pca_models_paths if 'aST' in x][0]
                basin_runoff_model = [x for x in pca_models_paths if 'basin_runoff' in x][0]
                thermal_forcing_model = [x for x in pca_models_paths if 'thermal_forcing' in x][0]
                    
                pca_models = dict(
                    aSMB=pkl.load(open(f"{self.pca_model_directory}/{aSMB_model}", "rb")),
                    aST=pkl.load(open(f"{self.pca_model_directory}/{aST_model}", "rb")),
                    basin_runoff=pkl.load(open(f"{self.pca_model_directory}/{basin_runoff_model}", "rb")),
                    thermal_forcing=pkl.load(open(f"{self.pca_model_directory}/{thermal_forcing_model}", "rb")),
                    sle=pkl.load(open(f"{self.pca_model_directory}/{sle_model}", "rb")),
                )
            else:
                pca_models = {}
                model_path = [x for x in pca_models_paths if var_name in x][0]
                pca_models[var_name] = pkl.load(open(f"{self.pca_model_directory}/{model_path}", "rb"))
            
        return pca_models
    
    def _load_scalers(self, scaler_directory, var_name='all'):
        if self.scaler_directory is None and scaler_directory is None:
            warnings.warn('self.scaler_directory is None, resorting to using self.pca_model_directory')
            if self.pca_model_directory is None:
                raise ValueError("Scaler directory must be specified, or DimensionalityReducer.generate_pca_models must be run first.")
            self.scaler_directory = self.pca_model_directory
        if scaler_directory is not None:
            self.scaler_directory = scaler_directory
        
        scaler_paths = os.listdir(self.scaler_directory)
        scaler_paths = [x for x in scaler_paths if 'scaler' in x and self.ice_sheet in x]
        
        if self.ice_sheet == 'AIS':
            
            if var_name not in ['all', 'evspsbl_anomaly', 'mrro_anomaly', 'pr_anomaly', 'smb_anomaly', 'ts_anomaly', 'thermal_forcing', 'salinity', 'temperature', 'sle', None]:
                raise ValueError(f"Variable name {var_name} not recognized.")
            
            if var_name == 'all' or var_name is None:
                evspsbl_model = [x for x in scaler_paths if 'evspsbl' in x][0]
                mrro_model = [x for x in scaler_paths if 'mrro' in x][0]
                pr_model = [x for x in scaler_paths if 'pr' in x][0]
                smb_model = [x for x in scaler_paths if 'smb' in x][0]
                ts_model = [x for x in scaler_paths if 'ts' in x][0]
                thermal_forcing_model = [x for x in scaler_paths if 'thermal_forcing' in x][0]
                salinity_model = [x for x in scaler_paths if 'salinity' in x][0]
                temperature_model = [x for x in scaler_paths if 'temperature' in x][0]
                sle_model = [x for x in scaler_paths if 'sle' in x][0]
                    
                scalers = dict(
                    evspsbl_anomaly=pkl.load(open(f"{self.scaler_directory}/{evspsbl_model}", "rb")),
                    mrro_anomaly=pkl.load(open(f"{self.scaler_directory}/{mrro_model}", "rb")),
                    pr_anomaly=pkl.load(open(f"{self.scaler_directory}/{pr_model}", "rb")),
                    smb_anomaly=pkl.load(open(f"{self.scaler_directory}/{smb_model}", "rb")),
                    ts_anomaly=pkl.load(open(f"{self.scaler_directory}/{ts_model}", "rb")),
                    thermal_forcing=pkl.load(open(f"{self.scaler_directory}/{thermal_forcing_model}", "rb")),
                    salinity=pkl.load(open(f"{self.scaler_directory}/{salinity_model}", "rb")),
                    temperature=pkl.load(open(f"{self.scaler_directory}/{temperature_model}", "rb")),
                    sle=pkl.load(open(f"{self.scaler_directory}/{sle_model}", "rb")),
                )
            else:
                scalers = {}
                scaler_path = [x for x in scaler_paths if var_name in x][0]
                scalers[var_name] = pkl.load(open(f"{self.scaler_directory}/{scaler_path}", "rb"))
        
        else: # GrIS
            if var_name not in ['all', 'aST', 'aSMB', 'basin_runoff', 'thermal_forcing', 'sle', None]:
                raise ValueError(f"Variable name {var_name} not recognized.")
            
            if var_name == 'all' or var_name is None:
                aSMB_model = [x for x in scaler_paths if 'aSMB' in x][0]
                aST_model = [x for x in scaler_paths if 'aST' in x][0]
                basin_runoff_model = [x for x in scaler_paths if 'basin_runoff' in x][0]
                thermal_forcing_model = [x for x in scaler_paths if 'thermal_forcing' in x][0]
                sle_model = [x for x in scaler_paths if 'sle' in x][0]
                    
                scalers = dict(
                    aSMB=pkl.load(open(f"{self.scaler_directory}/{aSMB_model}", "rb")),
                    aST=pkl.load(open(f"{self.scaler_directory}/{aST_model}", "rb")),
                    basin_runoff=pkl.load(open(f"{self.scaler_directory}/{basin_runoff_model}", "rb")),
                    thermal_forcing=pkl.load(open(f"{self.scaler_directory}/{thermal_forcing_model}", "rb")),
                    sle=pkl.load(open(f"{self.scaler_directory}/{sle_model}", "rb")),
                )
            else:
                scalers = {}
                scaler_path = [x for x in scaler_paths if var_name in x][0]
                scalers[var_name] = pkl.load(open(f"{self.scaler_directory}/{scaler_path}", "rb"))
    
        return scalers
    
    
    def transform(self, x, var_name, num_pcs=None, pca_model_directory=None, scaler_directory=None):
        """
        Transform the given variable into PCA space.

        Args:
            x (array-like): The input array containing the variables.
            variable (str): The name of the variable to transform.
            pca_models_paths (dict): A dictionary containing the filepaths for the PCA models.

        Returns:
            array-like: The transformed array.
        """
        if pca_model_directory is None and self.pca_model_directory is None:
            raise ValueError("PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first.")
        
        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory

        if scaler_directory is None and self.scaler_directory is None:
            raise ValueError("PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first.")
        
        if scaler_directory is not None:
            self.scaler_directory = scaler_directory
            
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
        
        pca_models = self._load_pca_models(self.pca_model_directory, var_name=var_name)
        scalers = self._load_scalers(self.scaler_directory, var_name=var_name)
        pca = pca_models[var_name]
        scaler = scalers[var_name]
        x = np.nan_to_num(x)
        scaled = scaler.transform(x)
        transformed = pca.transform(scaled)
        
        if num_pcs.endswith('%'):
            exp_var_pca = pca.explained_variance_ratio_
            cum_sum_eigenvalues = np.cumsum(exp_var_pca)
            num_pcs_cutoff = cum_sum_eigenvalues>float(num_pcs.replace('%', ""))/100
            if ~num_pcs_cutoff.any():
                warnings.warn(f'Explained variance cutoff ({num_pcs}) not reached, using all PCs available ({len(cum_sum_eigenvalues)}).')
                num_pcs = len(cum_sum_eigenvalues)
                
            else:
                num_pcs = np.argmax(num_pcs_cutoff)+1
            
        return transformed[:, :num_pcs]
    
    def invert(self, pca_x, var_name, pca_model_directory=None):
        """
        Invert the given variable from PCA space.

        Args:
            pca_x (array-like): The input array containing the variables in PCA space.
            variable (str): The name of the variable to transform.
            pca_models_paths (dict): A dictionary containing the filepaths for the PCA models.

        Returns:
            array-like: The inverted array.
        """
        if pca_model_directory is None and self.pca_model_directory is None:
            raise ValueError("PCA model directory must be specified, or DimensionalityReducer.generate_pca_models must be run first.")
        
        if pca_model_directory is not None:
            self.pca_model_directory = pca_model_directory
            
        pca_models = self._load_pca_models(pca_model_directory, var_name=var_name)
        pca = pca_models[var_name]
        inverted = pca.inverse_transform(pca_x)
        return inverted

    
            
def get_xarray_data(dataset_fp, var_name=None, ice_sheet='AIS'):
    """
    Retrieve a variable from an xarray dataset.

    Parameters:
    - dataset_fp (str): Filepath of the xarray dataset.
    - var_name (str): Name of the variable to retrieve.

    Returns:
    - data_flattened (ndarray): Flattened array of the variable values.
    - dataset (xarray.Dataset): Original dataset.

    """
    
    
    dataset = xr.open_dataset(dataset_fp, decode_times=False, engine='netcdf4')
    try:
        dataset = dataset.transpose('time', 'y', 'x', ...)
    except:
        pass
    
    # try dropping dimensions for atmospheric data
    if 'ivaf' in dataset.variables:
        pass
    
    else:
        
        # handle extra dimensions and variables
        try:
            dataset = dataset.drop_dims('nv4')
        except ValueError:
            pass
        
        for var in ['z_bnds', 'lat', 'lon', 'mapping', 'time_bounds']:
            try:
                dataset = dataset.drop(labels = [var])
            except ValueError:
                pass
        if 'z' in dataset.dims:
            dataset = dataset.mean(dim='z', skipna=True)
            
    
    if dataset.dims['x'] == 1681 and dataset.dims['y'] == 2881:
        dataset = dataset.sel(x=dataset.x.values[::5], y=dataset.y.values[::5])
    
    
    if var_name is not None:
        try:
            data = dataset[var_name].values
        except KeyError:
            return np.nan, np.nan
        
        x_dim = 761 if ice_sheet.lower() == 'ais' else 337
        y_dim = 761 if ice_sheet.lower() == 'ais' else 577
        if 'time' not in dataset.dims or dataset.dims['time'] == 1 or (data.shape[1] == y_dim and data.shape[2] == x_dim):
            pass
        else:
            # TODO: fix this. this is just a weird way of tranposing, not sure if it even happens.
            grid_indices = np.array([0, 1, 2])[(np.array(data.shape) == x_dim) | (np.array(data.shape) == y_dim)]
            data = np.moveaxis(data, list(grid_indices), [1, 2])
        
        if 'time' not in dataset.dims:
            data_flattened = data.reshape(-1,)
        else:
            data_flattened = data.reshape(len(dataset.time), -1)
        return data_flattened
    
    return dataset


class DatasetMerger:
    def __init__(self, ice_sheet, forcing_dir, projection_dir, experiment_file, output_dir):
        self.ice_sheet = ice_sheet
        self.forcing_dir = forcing_dir
        self.projection_dir = projection_dir
        self.experiment_file = experiment_file
        self.output_dir = output_dir
        
        if self.experiment_file.endswith('.csv'):
            self.experiments = pd.read_csv(experiment_file)
        elif self.experiment_file.endswith('.json'):
            self.experiments = pd.read_json(experiment_file).T
        else:
            raise ValueError("Experiment file must be a CSV or JSON file.")
        
        self.forcing_paths = get_all_filepaths(path=self.forcing_dir, filetype='csv',)
        self.projection_paths = get_all_filepaths(path=self.projection_dir, filetype='csv',)
        self.forcing_metadata = self._get_forcing_metadata()
        

    def merge_dataset(self, ):
        full_dataset = pd.DataFrame()
        for projection in self.projection_paths:
            # get experiment from projection filepath
            exp = projection.replace('.csv', '').split('/')[-1].split('_')[-1]
            
            # make sure cases match when doing table lookup
            self.experiments['exp'] = self.experiments['exp'].apply(lambda x: x.lower())
            
            # get AOGCM value from table lookup
            aogcm = self.experiments.loc[self.experiments.exp == exp.lower()]['AOGCM'].values[0]
            proj_cmip_model = aogcm.split('_')[0]
            proj_pathway = aogcm.split('_')[-1]
            
            # get forcing file from table lookup that matches projection
            try:
                forcing_file = self.forcing_metadata.file.loc[(self.forcing_metadata.cmip_model == proj_cmip_model) & (self.forcing_metadata.pathway == proj_pathway)].values[0]
            except IndexError:
                raise IndexError(f"Could not find forcing file for {aogcm}. Check formatting of experiment file.")
            
            # load forcing and projection datasets
            forcings = pd.read_csv(f"{self.forcing_dir}/{forcing_file}.csv")
            projections = pd.read_csv(projection)
            # if forcings are longer than projections, cut off the beginning of the forcings
            if len(forcings > len(projections)):
                forcings = forcings.iloc[-len(projections):].reset_index(drop=True)
                
            # add forcings and projections together and add some metadata
            merged_dataset = pd.concat([forcings, projections], axis=1)
            merged_dataset['cmip_model'] = proj_cmip_model
            merged_dataset['pathway'] = proj_pathway
            merged_dataset['exp'] = exp
            
            # now add to dataset with all forcing/projection pairs
            full_dataset = pd.concat([full_dataset, merged_dataset]) 
            
        # save the full dataset
        full_dataset.to_csv(f"{self.output_dir}/dataset.csv", index=False)
        
        return 0
    

    def _get_forcing_metadata(self, ):
        pairs = {}
        # loop through forcings, looking for cmip model and pathway
        for forcing in self.forcing_paths:
            forcing = forcing.replace('.csv', '').split('/')[-1]
            cmip_model = forcing.split('_')[1]
            
            if 'rcp' in forcing or 'ssp' in forcing.lower():
                for substring in forcing.split('_'):
                    if 'rcp' in substring or 'ssp' in substring:
                        pathway = substring.lower()
                        break
            else:
                pathway = 'rcp85'
            pairs[forcing] = [cmip_model.lower(), pathway.lower()]
        df = pd.DataFrame(pairs).T
        df = pd.DataFrame(pairs).T.reset_index()
        df.columns = ['file', 'cmip_model', 'pathway']
            
        return df
    
def combine_gris_forcings(forcing_dir,):
    atmosphere_dir = f"{forcing_dir}/GrIS/Atmosphere_Forcing/aSMB_observed/v1/"
    cmip_directories = next(os.walk(atmosphere_dir))[1]
    cmip_loop = tqdm(cmip_directories, total=len(cmip_directories))
    for cmip_dir in cmip_loop:
        cmip_loop.set_description(f"Processing {cmip_dir}")
        for var in [f"aSMB", f"aST"]:
            files = os.listdir(f"{atmosphere_dir}/{cmip_dir}/{var}")
            files = np.array([x for x in files if x.endswith('.nc')])
            years = np.array([int(x.replace('.nc', "").split('-')[-1]) for x in files])
            year_files = files[(years >= 2015) & (years <= 2100)]
            

            for i, file in enumerate(year_files,):
                # first iteration, open dataset and store
                if i == 0:
                    dataset = xr.open_dataset(f"{atmosphere_dir}/{cmip_dir}/{var}/{file}")
                    for dim in ['nv', 'nv4', 'mapping']:
                            try:
                                dataset = dataset.drop_dims(dim)
                            except:
                                pass
                    dataset = dataset.drop('mapping')
                    dataset = dataset.sel(x=dataset.x.values[::5], y=dataset.y.values[::5])
                    continue
                
                # following iterations, open dataset and concatenate
                data = xr.open_dataset(f"{atmosphere_dir}/{cmip_dir}/{var}/{file}")
                for dim in ['nv', 'nv4', ]:
                    try:
                        data = data.drop_dims(dim)
                    except:
                        pass
                data = data.drop('mapping')
                data = data.sel(x=data.x.values[::5], y=data.y.values[::5])
                # data['time'] = pd.to_datetime(year, format='%Y')
                dataset = xr.concat([dataset, data], dim='time')

            # Now you have the dataset with the files loaded and time dimension set
            dataset.to_netcdf(os.path.join(atmosphere_dir, cmip_dir, f"GrIS-{cmip_dir}-{var}-combined.nc"))

    return 0

            
    
        