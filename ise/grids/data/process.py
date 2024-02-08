import xarray as xr
import os
import pandas as pd
import numpy as np
import warnings
import cftime
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
from ise.grids.utils import get_all_filepaths
from datetime import date, timedelta, datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
# warnings.simplefilter("ignore") 



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
    
    def process(self, output_directory):
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
        
        if output_directory is None:
            raise ValueError("Output directory must be specified")
        
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
        
        # directory = r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-AIS/ILTS_PIK/SICOPOLIS/exp05"
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
        bed = xr.open_dataset(os.path.join(directory, f'topg_{naming_convention}'), decode_times=False).transpose('x', 'y', 'time', ...)
        thickness = xr.open_dataset(os.path.join(directory, f'lithk_{naming_convention}'), decode_times=False).transpose('x', 'y', 'time', ...)
        mask = xr.open_dataset(os.path.join(directory, f'sftgif_{naming_convention}'), decode_times=False).transpose('x', 'y', 'time', ...)
        ground_mask = xr.open_dataset(os.path.join(directory, f'sftgrf_{naming_convention}'), decode_times=False).transpose('x', 'y', 'time', ...)
        length_time = len(thickness.time)
        # note on decode_times=False -- by doing so, it stays in "days from" rather than trying to infer a type. Makes handling much more predictable.


        # if -9999 instead of np.nan, replace (come back and optimize? couldn't figure out with xarray)
        if bed.topg[0,0,0] <= -9999. or bed.topg[0,0,0] >= 9999:
            topg = bed.topg.values
            topg[(np.where(topg <= -9999.)) | (np.where(topg >= 9999))] = np.nan
            bed['topg'].values = topg
            del topg
            
            lithk = thickness.lithk.values
            lithk[(np.where(lithk <= -9999.)) | (np.where(lithk >= 9999))] = np.nan
            thickness['lithk'].values = lithk
            del lithk
            
            sftgif = mask.sftgif.values
            sftgif[(np.where(sftgif <= -9999.)) | (np.where(sftgif >= 9999))] = np.nan
            mask['sftgif'].values = sftgif
            del sftgif
            
            sftgrf = ground_mask.sftgrf.values
            sftgrf[(np.where(sftgrf <= -9999.)) | (np.where(sftgrf >= 9999))] = np.nan
            ground_mask['sftgrf'].values = sftgrf
            del sftgrf
        
        # converts time (in "days from X" to numpy.datetime64) and subsets time from 2015 to 2100
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
        if 'time' not in bed.dims or bed.dims['time'] == 1:
            try:
                bed = bed.drop_vars(['time',])
            except ValueError:
                pass
            bed = bed.expand_dims(dim={'time': length_time})
            
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
    elif isinstance(dataset.time.values[0], np.float32) or isinstance(dataset.time.values[0], np.float64):
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
        else:
            try:
                start_date = np.datetime64(datetime.strptime(units.replace("days since ", ""), "%Y-%m-%d"))
            except ValueError:
                start_date = np.datetime64(datetime.strptime(units.replace("days since ", ""), "%d-%m-%Y"))

            dataset['time'] = np.array([start_date + np.timedelta64(int(x), 'D') for x in dataset.time.values])
    if len(dataset.time) > 86:
        # make sure the max date is 2100
        # dataset = dataset.sel(time=slice(np.datetime64('2014-01-01'), np.datetime64('2101-01-01')))
        dataset = dataset.sel(time=slice('2014-01-01', '2101-01-01'))
        
        # if you still have more than 86, take the previous 86 values from 2100
        if len(dataset.time) > 86:
            # LSCE GRISLI has two 2015 measurements
            dataset = dataset.sel(time=slice(dataset.time.values[len(dataset.time) - 86], dataset.time.values[-1]))
        
    if len(dataset.time) != 86:
        raise ValueError('Subsetting of NC file processed incorrectly, go back and check logs.')
            
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
    def __init__(self, forcing_dir, projection_dir, output_dir, nan_mask=None):
        super().__init__()
        if forcing_dir is None:
            raise ValueError("Forcing directory must be specified.")
        if output_dir is None:
            raise ValueError("Output directory must be specified.")
        self.forcing_dir = forcing_dir
        self.projection_dir = projection_dir
        self.output_dir = output_dir
        self.forcing_paths = {'all': None, 'atmosphere': None, 'ocean': None}
        self.pca_model_directory = None
        
        all_forcing_fps = get_all_filepaths(path=self.forcing_dir, filetype='nc', contains='1995-2100', not_contains='Ice_Shelf_Fracture')
        self.forcing_paths['all'] = [x for x in all_forcing_fps if '8km' in x and 'v1' not in x]
        self.forcing_paths['atmosphere'] = [x for x in self.forcing_paths['all'] if "Atmosphere_Forcing" in x]
        self.forcing_paths['ocean'] = [x for x in self.forcing_paths['all'] if "Ocean_Forcing" in x]
        
        all_projection_fps = get_all_filepaths(path=self.projection_dir, filetype='nc', contains='ivaf', not_contains='ctrl_proj')
        self.projection_paths = all_projection_fps
        
        if nan_mask is None:
            warnings.warning('Could not find nan_mask, running generate_nan_mask()')
            self.nan_mask = self.generate_nan_mask(paths=self.forcing_paths['all'].extend(self.projection_paths))
        elif isinstance(nan_mask, str):
            if nan_mask.endswith('.csv'):
                self.nan_mask = pd.read_csv(nan_mask).to_numpy()
            else:
                raise NotImplementedError('Only supported nan_mask filetype is CSV.')
        elif isinstance(nan_mask, np.array):
            self.nan_mask = nan_mask
        else:
            raise TypeError('nan_mask can only be path (str), None (generate), or numpy array (np.array).')
        
        self.nan_mask_args = np.argwhere(self.nan_mask.flatten() == 0).squeeze() # nan_mask = 0 means no nan values found there
                
        
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
        
        # Train PCA models for each atmospheric and oceanic forcing variable and save
        # self._generate_atmosphere_pcas(self.forcing_paths['atmosphere'], self.pca_model_directory)
        # self._generate_ocean_pcas(self.forcing_paths['ocean'], self.pca_model_directory)
        
        # Train PCA model for SLE and save
        sle_paths = get_all_filepaths(path=self.projection_dir, filetype='nc', contains='ivaf', not_contains='ctrl')
        self._generate_sle_pca(sle_paths, save_dir=self.pca_model_directory)
        
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
        for i, path in enumerate(self.forcing_paths['atmosphere']):
            print(f"{i}/{len(self.forcing_paths['atmosphere'])} atmospheric forcing files converted to PCA space.")
            dataset = xr.open_dataset(path, decode_times=False, engine='netcdf4').transpose('x', 'y', 'time', ...)  # open the dataset
            forcing_name = path.replace('.nc', '').split('/')[-1]  # get metadata (model, ssp, etc.)
            
            # transform each variable in the dataset with their respective trained PCA model
            transformed_data = {}
            for var in ['evspsbl_anomaly', 'mrro_anomaly', 'pr_anomaly', 'smb_anomaly', 'ts_anomaly']:
                try:
                    transformed = self.transform(dataset[var].values, var_name=var, num_pcs=num_pcs, pca_model_directory=self.pca_model_directory)
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
        
        # for each atmospheric forcing file, convert each variable to PCA space with pretrained PCA model
        for i, path in enumerate(self.forcing_paths['ocean']):
            
            # open the dataset
            print(f"{i}/{len(self.forcing_paths['ocean'])} oceanic forcing files converted to PCA space.")
            forcing_name = path.replace('.nc', '').split('/')[-1]  # get metadata (model, ssp, etc.)
            
            # get variable name by splitting the filepath name
            var = self.forcing_paths['ocean'][i].split('/')[-1].split('_')[-4]
            if var == 'forcing' or var == 'thermal':
                var = 'thermal_forcing'
            
            # get forcing array (requires mean value over z dimensions, see get_xarray_variable())
            forcing_array, _ = get_xarray_variable(path, var_name=var)
            # forcing_array = np.nan_to_num(forcing_array)  # deal with np.nans
            # forcing_array = forcing_array[:, ~(np.isnan(forcing_array).any(axis=0))]
            
            # transform each variable in the dataset with their respective trained PCA model
            transformed_data = {}
            transformed = self.transform(forcing_array, var_name=var, num_pcs=num_pcs, pca_model_directory=self.pca_model_directory)
            transformed_data[var] = transformed  # store in dict with structure {'var_name': transformed_var}
            
            # create a dataframe with rows corresponding to time (106 total) and columns corresponding to each variables principal components
            variable_df = pd.DataFrame(transformed_data[var], columns=[f"{var}_pc{i+1}" for i in range(transformed_data[var].shape[1])])
            variable_df.to_csv(f"{output_dir}/forcings/PCA_{forcing_name}.csv", index=False)
            
            
        print(f"{len(self.forcing_paths['ocean'])}/{len(self.forcing_paths['ocean'])} oceanic forcing files converted to PCA space.")
        print(f'Finished converting oceanic forcings to PCA space, files outputted to {output_dir}.')
        
                
        return 0
        
        
            
    # def convert_outputs()
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
        for i, path in enumerate(self.projection_paths):
            print(f"{i}/{len(self.projection_paths)} projection files converted to PCA space.")
            
            # get forcing array (requires mean value over z dimensions, see get_xarray_variable())
            projection_array, _ = get_xarray_variable(path, var_name='ivaf')
            # nan_indices = np.argwhere(np.isnan(projection_array))
            # print(len(nan_indices))
            # continue
            
            # projection_array = np.nan_to_num(projection_array)  # deal with np.nans
            projection_array = projection_array / 1e9 / 362.5
            var = 'sle'  
            # projection_array = np.nan_to_num(projection_array)  # there shouldn't be nans...
            projection_name = path.replace('.nc', '').split('/')[-1]  # get metadata (model, ssp, etc.)
            
            # transform each variable in the dataset with their respective trained PCA model
            transformed_data = {}
            transformed = self.transform(projection_array, var_name=var, num_pcs=num_pcs, pca_model_directory=self.pca_model_directory)
            transformed_data[var] = transformed  # store in dict with structure {'var_name': transformed_var}
            
            # create a dataframe with rows corresponding to time (106 total) and columns corresponding to each variables principal components
            variable_df = pd.DataFrame(transformed_data[var], columns=[f"{var}_pc{i+1}" for i in range(transformed_data[var].shape[1])])
            variable_df.to_csv(f"{output_dir}/projections/PCA_{projection_name}.csv", index=False)
        
        print(f"{len(self.projection_paths)}/{len(self.projection_paths)} projection files converted to PCA space.")
        print(f'Finished converting projections to PCA space, files outputted to {output_dir}.')
    
    

    def _generate_atmosphere_pcas(self, atmosphere_fps: list, save_dir: str):
        """
        Generate principal component analysis (PCA) for atmospheric variables.

        Args:
            atmosphere_fps (list): List of file paths to atmospheric CMIP files.
            save_dir (str): Directory to save the PCA results.

        Returns:
            int: 0 if successful.
        """
        
        # for each variable
        var_names = ['pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly', 'smb_anomaly', 'ts_anomaly']
        for i, var in enumerate(var_names):
            print(f"{i}/{len(var_names)} atmospheric pca models created.")
            variable_array = np.zeros([len(atmosphere_fps), 106, 761*761])
            
            # loop through each atmospheric CMIP file
            for i, fp in enumerate(atmosphere_fps):
                
                # get the variable you need (rather than the entire dataset)
                data_flattened, dataset = get_xarray_variable(fp, var)
                
                # store it in the total array
                variable_array[i, :, :] = data_flattened     
                
            
            # deal with np.nans (ask about later) -- since it's an anomaly, replace with 0
            variable_array = np.nan_to_num(variable_array)   
            
            # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
            variable_array = variable_array.reshape(len(atmosphere_fps)*len(dataset.time), 761*761)
            
            # run PCA
            # if np.isnan(variable_array).any():
            #     continue
            pca, _ = self._run_PCA(variable_array, num_pcs=300)
            
            # change back to (num_files, num_timestamps, num_pcs)
            # pca_array = pca_array.reshape(len(atmosphere_fps), len(dataset.time), -1)
            
            # get percent explained
            exp_var_pca = pca.explained_variance_ratio_
            cum_sum_eigenvalues = np.cumsum(exp_var_pca)
            arg_90 = np.argmax(cum_sum_eigenvalues>0.90)+1
            print(f"Variable {var} has {arg_90} PCs that explain 90% of the variance")
            
            # output pca object
            save_path = f"{save_dir}/pca_{var}_{arg_90}pcs.pkl"
            pkl.dump(pca, open(save_path,"wb"))
            
        return 0
    
    def _generate_ocean_pcas(self, ocean_fps: list, save_dir: str):
        """
        Generate principal component analysis (PCA) for ocean variables.

        Args:
            ocean_fps (list): List of file paths for ocean variables.
            save_dir (str): Directory to save the PCA results.

        Returns:
            int: 0 if PCA generation is successful, -1 otherwise.
        """
        
        thermal_forcing_fps = [x for x in ocean_fps if 'thermal_forcing' in x]
        salinity_fps = [x for x in ocean_fps if 'salinity' in x]
        tempereature_fps = [x for x in ocean_fps if 'temperature' in x]
        
        thermal_forcing_array = np.zeros([len(thermal_forcing_fps), 106, 761*761])
        salinity_array = np.zeros([len(salinity_fps), 106, 761*761])
        temperature_array = np.zeros([len(tempereature_fps), 106, 761*761])
        
        # get the variables you need (rather than the entire dataset)
        for i, fp in enumerate(thermal_forcing_fps):
            data_flattened, dataset = get_xarray_variable(fp, var_name='thermal_forcing')
            thermal_forcing_array[i, :, :] = data_flattened # store
        for i, fp in enumerate(salinity_fps):
            data_flattened, dataset = get_xarray_variable(fp, var_name='salinity')
            salinity_array[i, :, :] = data_flattened # store
        for i, fp in enumerate(tempereature_fps):
            data_flattened, dataset = get_xarray_variable(fp, var_name='temperature')
            temperature_array[i, :, :] = data_flattened
            
        # thermal_forcing_array = thermal_forcing_array[~np.isnan(thermal_forcing_array)]
        # salinity_array = salinity_array[~np.isnan(salinity_array)]
        # temperature_array = temperature_array[~np.isnan(temperature_array)]
        
        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        thermal_forcing_array = thermal_forcing_array.reshape(len(thermal_forcing_fps)*len(dataset.time), 761*761)
        salinity_array = salinity_array.reshape(len(salinity_fps)*len(dataset.time), 761*761)
        temperature_array = temperature_array.reshape(len(tempereature_fps)*len(dataset.time), 761*761)
        
        # remove nans
        # thermal_forcing_array = thermal_forcing_array[:, ~(np.isnan(thermal_forcing_array).any(axis=0))]
        # salinity_array = salinity_array[:, ~(np.isnan(salinity_array).any(axis=0))]
        # temperature_array = temperature_array[:, ~(np.isnan(temperature_array).any(axis=0))]
        thermal_forcing_array = np.nan_to_num(thermal_forcing_array)
        salinity_array = np.nan_to_num(salinity_array)
        temperature_array = np.nan_to_num(temperature_array)
        
        
        
        # run PCA
        pca_tf, _ = self._run_PCA(thermal_forcing_array, num_pcs=300)
        pca_sal, _ = self._run_PCA(salinity_array, num_pcs=300)
        pca_temp, _ = self._run_PCA(temperature_array, num_pcs=300)
        
        # get percent explained
        tf_exp_var_pca = pca_tf.explained_variance_ratio_
        tf_cum_sum_eigenvalues = np.cumsum(tf_exp_var_pca)
        tf_arg_90 = np.argmax(tf_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/pca_thermal_forcing_{tf_arg_90}pcs.pkl"
        pkl.dump(pca_tf, open(save_path,"wb"))
        # np.save(f"{save_dir}/data/thermal_forcing_{tf_arg_90}pcs.npy", pca_tf_array)
        
        sal_exp_var_pca = pca_sal.explained_variance_ratio_
        sal_cum_sum_eigenvalues = np.cumsum(sal_exp_var_pca)
        sal_arg_90 = np.argmax(sal_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/pca_salinity_{sal_arg_90}pcs.pkl"
        pkl.dump(pca_sal, open(save_path,"wb"))
        # np.save(f"{save_dir}/data/salinity_{sal_arg_90}pcs.npy", pca_sal_array)
        
        temp_exp_var_pca = pca_temp.explained_variance_ratio_
        temp_cum_sum_eigenvalues = np.cumsum(temp_exp_var_pca)
        temp_arg_90 = np.argmax(temp_cum_sum_eigenvalues>0.90)+1
        save_path = f"{save_dir}/pca_temperature_{temp_arg_90}pcs.pkl"
        pkl.dump(pca_temp, open(save_path,"wb"))
        # np.save(f"{save_dir}/data/temperature_{temp_arg_90}pcs.npy", pca_temp_array)
        
        return 0
    
    def _generate_sle_pca(self, sle_fps: list, save_dir: str):
        """
        Generate principal component analysis (PCA) for sea level equivalent (SLE) variables.

        Args:
            sle_fps (list): List of file paths for SLE variables.
            save_dir (str): Directory to save the PCA results.

        Returns:
            int: 0 if PCA generation is successful, -1 otherwise.
        """
        
        sle_array = np.zeros([4, 86, 761*761])
        
        # loop through each SLE (IVAF) projection file
        for i, fp in enumerate(sle_fps):
            
            # get the variable
            try:
                data_flattened, _ = get_xarray_variable(fp, var_name="sle")
            except:
                data_flattened, _ = get_xarray_variable(fp, var_name="ivaf")
                data_flattened = data_flattened / 1e9 / 362.5
                
            
            # if data_flattened.shape[0] > 86:
            #     data_flattened = data_flattened[-86:,:]
            
            # store it in the total array
            sle_array[i, :, :] = data_flattened 
            
            if i == 3:
                sle_fps = sle_fps[0:4]
                break
                
            
        
        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        sle_array = sle_array.reshape(len(sle_fps)*86, 761*761)
        
        # only keep values within the mask
        # sle_array = sle_array[:, self.nan_mask_args]
        
        
        # since the array is so large (350*85, 761*761) = (29750, 579121), randomly sample N rows and run PCA
        sle_array = sle_array[np.random.choice(sle_array.shape[0], 300, replace=False), :]
        
        
        # deal with np.nans (ask about later)
        sle_array = np.nan_to_num(sle_array) 
        
        # normalize sle
        # sle_array = MinMaxScaler().fit_transform(sle_array)
        # sle_array = (sle_array - np.min(sle_array)) / (np.max(sle_array) - np.min(sle_array))
        
        # run pca
        pca, _ = self._run_PCA(sle_array, num_pcs=300,)
        
        
        # get percent explained
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        arg_90 = np.argmax(cum_sum_eigenvalues>0.90)+1
        print(f"Variable SLE has {arg_90} PCs that explain 90% of the variance")
        
        # output pca object
        save_path = f"{save_dir}/pca_sle_{arg_90}pcs.pkl"
        pkl.dump(pca, open(save_path,"wb"))
        
        
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
        if var_name not in ['all', 'evspsbl_anomaly', 'mrro_anomaly', 'pr_anomaly', 'smb_anomaly', 'ts_anomaly', 'thermal_forcing', 'salinity', 'temperature', 'sle', None]:
            raise ValueError(f"Variable name {var_name} not recognized.")
            
        pca_models_paths = os.listdir(self.pca_model_directory)
        pca_models_paths = [x for x in pca_models_paths if 'pca' in x]
        
        if var_name == 'all' or var_name is None:
            evspsbl_model = [x for x in pca_models_paths if 'evspsbl' in x][0]
            mrro_model = [x for x in pca_models_paths if 'mrro' in x][0]
            pr_model = [x for x in pca_models_paths if 'pr' in x][0]
            smb_model = [x for x in pca_models_paths if 'smb' in x][0]
            ts_model = [x for x in pca_models_paths if 'ts' in x][0]
            thermal_forcing_model = [x for x in pca_models_paths if 'thermal_forcing' in x][0]
            salinity_model = [x for x in pca_models_paths if 'salinity' in x][0]
            temperature_model = [x for x in pca_models_paths if 'temperature' in x][0]
                
            pca_models = dict(
                evspsbl_anomaly=pkl.load(open(f"{self.pca_model_directory}/{evspsbl_model}", "rb")),
                mrro_anomaly=pkl.load(open(f"{self.pca_model_directory}/{mrro_model}", "rb")),
                pr_anomaly=pkl.load(open(f"{self.pca_model_directory}/{pr_model}", "rb")),
                smb_anomaly=pkl.load(open(f"{self.pca_model_directory}/{smb_model}", "rb")),
                ts_anomaly=pkl.load(open(f"{self.pca_model_directory}/{ts_model}", "rb")),
                thermal_forcing=pkl.load(open(f"{self.pca_model_directory}/{thermal_forcing_model}", "rb")),
                salinity=pkl.load(open(f"{self.pca_model_directory}/{salinity_model}", "rb")),
                temperature=pkl.load(open(f"{self.pca_model_directory}/{temperature_model}", "rb")),
            )
        else:
            pca_models = {}
            model_path = [x for x in pca_models_paths if var_name in x][0]
            pca_models[var_name] = pkl.load(open(f"{self.pca_model_directory}/{model_path}", "rb"))
        return pca_models
    
    def transform(self, x, var_name, num_pcs=None, pca_model_directory=None):
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
            
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
            
        nan_mask = np.isnan(x)
        x = np.nan_to_num(x)
        
        pca_models = self._load_pca_models(pca_model_directory, var_name=var_name)
        pca = pca_models[var_name]
        transformed = pca.transform(x)
        
        if num_pcs.endswith('%'):
            exp_var_pca = pca.explained_variance_ratio_
            cum_sum_eigenvalues = np.cumsum(exp_var_pca)
            num_pcs = np.argmax(cum_sum_eigenvalues>float(num_pcs.replace('%', ""))/100)+1
            
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
    
def generate_nan_mask(self, paths, output_dir=None):
    # for each file, loop through data and find where nan values are
    for i, fp in tqdm(enumerate(paths), total=len(paths)):
        d = xr.open_dataset(fp, decode_times=True)
        d = d.transpose('x', 'y', 'time', ...)
        
        # decide which variable to pull out based on file type
        ocean = False
        if 'Atmosphere' in fp:
            var = 'smb_anomaly'

        elif 'Ocean' in fp:
            ocean = True
            var = fp.split('/')[-1].split('_')[-4]
            if var == 'forcing':
                var = 'thermal_forcing'
        else:
            var = 'ivaf'
        
        # on the first file, first iteration, keep the first 761*761 as the mask
        if i == 0:
            if ocean:
                mask = np.isnan(d[var][:, :, 0, 0]) * 1 # ocean data is [x, y, time, z]
            else:
                mask = np.isnan(d[var][:, :, 0]) * 1
        
        # loop through each sequential mask and only keep the minimum number of nan values
        for j in range(len(d.time)):
            if ocean:
                nan_index_new = np.isnan(d[var][:, :, j, 0]) * 1
            else:
                nan_index_new = np.isnan(d[var][:, :, j]) * 1
            # if an argument is np.nan for all data, the value of mask will stay 1, else 0
            mask = np.multiply(nan_index_new, mask) 
        
        del d # delete old dataset before loading next
    
    if output_dir is not None:
        np.savetxt(f"{output_dir}/nan_mask.csv", mask, delimiter=",")
    else:
        np.savetxt(f"nan_mask.csv", mask, delimiter=",")
    
    return mask
        


    
            
def get_xarray_variable(dataset_fp, var_name):
    """
    Retrieve a variable from an xarray dataset.

    Parameters:
    - dataset_fp (str): Filepath of the xarray dataset.
    - var_name (str): Name of the variable to retrieve.

    Returns:
    - data_flattened (ndarray): Flattened array of the variable values.
    - dataset (xarray.Dataset): Original dataset.

    """
    
    try:
        dataset = xr.open_dataset(dataset_fp, decode_times=True, engine='netcdf4')
    except ValueError:
        dataset = xr.open_dataset(dataset_fp, decode_times=False, engine='netcdf4')
    
    # try dropping dimensions for atmospheric data
    if 'ivaf' in dataset.variables:
        pass
    else:
        try:
            dataset = dataset.drop_dims('nv4')
            dataset = dataset.drop(['lat2d', 'lon2d'])
        except ValueError:
            # if its oceanic data...
            try:
                dataset = dataset.drop(labels=["z_bnds", "lat", "lon"])
            except ValueError:
                dataset = dataset.drop(labels=['z_bnds'])
                
            dataset = dataset.mean(dim='z', skipna=True)

    try:
        data = dataset[var_name].values
    except KeyError:
        return np.nan, np.nan
    
    if data.shape[1] == 761 and data.shape[2] == 761:
        pass
    else:
        grid_indices = np.array([0, 1, 2])[np.array(data.shape) == 761]
        data = np.moveaxis(data, list(grid_indices), [1, 2])
        
    data_flattened = data.reshape(len(dataset.time), -1)
    
    return data_flattened, dataset


class DatasetMerger:
    def __init__(self, forcing_dir, projection_dir, experiment_file, output_dir):
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

            
    
        