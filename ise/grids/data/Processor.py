import xarray as xr
import os
import pandas as pd
import numpy as np
import warnings
import cftime
from sklearn.decomposition import PCA
import pickle as pkl


class Processor:
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
        
    def process_forcings(self, output_directory):
        if self.forcings_directory is None:
            raise ValueError("Forcings path must be specified")
        
        
        # generate PCA models for each forcing
        all_forcing_fps = get_all_filepaths(path=self.forcing_directory, filetype='nc', contains='1995-2100', not_contains='Ice_Shelf_Fracture')
        forcing_fps = [x for x in all_forcing_fps if '8km' in x and 'v1' not in x]
        atmosphere_fps = [x for x in forcing_fps if "Atmosphere_Forcing" in x]
        ocean_fps = [x for x in forcing_fps if "Ocean_Forcing" in x]
        
        generate_atmosphere_pcas(atmosphere_fps, output_directory)
        generate_ocean_pcas(ocean_fps, output_directory)
        
        
        pass
    
    def process_projections(self, output_directory):
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
        
        
        # get ivaf files in projections directory
        ivaf_files = []
        for root, dirs, files in os.walk(self.projections_directory):
            for file in files:
                if 'ivaf' in file and 'ctrl_proj' not in file:
                    ivaf_files.append(os.path.join(root, file))
        
        # create array of ivaf values of shape (num_files, x, y, time) and store in projections_array
        if self.ice_sheet.upper() == "AIS":
            projections_array = np.zeros([len(ivaf_files), 761, 761, 86])
        else:
            projections_array = np.zeros([len(ivaf_files), 337, 577, 86])
            
        # load individual ivaf file arrays into a single array (num_files, x, y, time)
        metadata = []
        for i, file in enumerate(ivaf_files):
            try:
                data = xr.open_dataset(file)
            except ValueError:
                data = xr.open_dataset(file, decode_times=False)
            ivaf = data.ivaf.values
            if ivaf.shape[2] != 86:
                ivaf = ivaf[:,:,0:86]
            projections_array[i, :, :, :] = ivaf
            
            # capture metadata for storage
            path = file.split('/')
            exp = path[-2]
            model = path[-3]
            group = path[-4]
            metadata.append([group, model, exp])
            
        # save projections array and metadata
        np.save(f"{output_directory}/projections_{self.ice_sheet}.npy", projections_array)
        metadata_df = pd.DataFrame(metadata, columns=['group', 'model', 'exp'])
        metadata_df.to_csv(f"{output_directory}/projections_{self.ice_sheet}_metadata.csv", index=False)
            
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
        
        ctrl_proj_dirs = []
        exp_dirs = []
        for root, dirs, files in os.walk(data_directory):
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
        for directory in exp_dirs: # [print(x, i) for x, i in enumerate(exp_dirs)]
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
        bed = xr.open_dataset(os.path.join(directory, f'topg_{naming_convention}'), decode_times=False)
        thickness = xr.open_dataset(os.path.join(directory, f'lithk_{naming_convention}'), decode_times=False)
        mask = xr.open_dataset(os.path.join(directory, f'sftgif_{naming_convention}'), decode_times=False)
        ground_mask = xr.open_dataset(os.path.join(directory, f'sftgrf_{naming_convention}'), decode_times=False)
        length_time = len(thickness.time)
        
        
        #! TODO: Ask about this -- idk if it really matters? x/y doesn't get used regardless
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
        bed_data = np.transpose(bed.topg.values, (2,1,0))
        thickness_data = np.transpose(thickness.lithk.values, (2,1,0))
        mask_data = np.transpose(mask.sftgif.values, (2,1,0))
        ground_mask_data = np.transpose(ground_mask.sftgrf.values, (2,1,0))
        
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
            try:
                ivaf_ctrl = xr.open_dataset(ctrl_path, )
            except ValueError:
                ivaf_ctrl = xr.open_dataset(ctrl_path, decode_times=False)
            
            # some include historical values (going back to 2005 for example), subset those out
            if ivaf_ctrl.time.values.shape[0] > 87:
                if isinstance(bed.time.values[0], cftime._cftime.DatetimeNoLeap):
                    datetimeindex = ivaf_ctrl.indexes['time'].to_datetimeindex()
                    ivaf_ctrl['time'] = datetimeindex
                if isinstance(ivaf_ctrl.time.values[0], cftime._cftime.DatetimeNoLeap):
                    datetimeindex = ivaf_ctrl.indexes['time'].to_datetimeindex()
                    ivaf_ctrl['time'] = datetimeindex
                # elif isinstance(bed.time.values[0])
                ivaf_ctrl = ivaf_ctrl.sel(time=slice(np.datetime64('2015-01-01'), np.datetime64('2101-01-01')))
            
            # if the time lengths don't match (one goes for 87 years and the other 86) select only time frames that match
            if ivaf_ctrl.time.values.shape[0] > ivaf.shape[2]:
                ivaf_ctrl = ivaf_ctrl.isel(time=slice(0,ivaf.shape[2]))
                ivaf_nc = ivaf_nc.drop_sel(time=ivaf_nc.time.values[:(ivaf.shape[2]-ivaf_ctrl.time.values.shape[0])]) # drop extra time steps
            elif ivaf_ctrl.time.values.shape[0] < ivaf.shape[2]:
                ivaf_nc = ivaf_nc.drop_sel(time=ivaf_nc.time.values[ivaf_ctrl.time.values.shape[0]-ivaf.shape[2]:]) # drop extra time steps
                ivaf = ivaf[:,:, 0:ivaf_ctrl.time.values.shape[0]]
                
            else:
                pass
                
            ivaf = ivaf_ctrl.ivaf.values - ivaf
        else:
            pass
            
        # save ivaf file
        ivaf_nc['ivaf'] = (('x', 'y', 'time'), ivaf)
        ivaf_nc = ivaf_nc.drop_vars(['topg',])
        ivaf_nc.to_netcdf(os.path.join(directory, f'ivaf_{self.ice_sheet}_{group}_{model}_{exp}.nc'))

        print(f"{group}_{model}_{exp}: Processing successful.")
        
        return 1
        
        

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
                    dataset = xr.open_dataset(file_path, decode_times=False)

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

def get_all_filepaths(path: str, filetype: str = None, contains: str = None, not_contains: str = None):
    """Retrieves all filepaths for files within a directory. Supports subsetting based on filetype
    and substring search.

    Args:
        path (str): Path to directory to be searched.
        filetype (str, optional): File type to be returned (e.g. csv, nc). Defaults to None.
        contains (str, optional): Substring that files found must contain. Defaults to None.
        not_contains(str, optional): Substring that files found must NOT contain. Defaults to None.

    Returns:
        List[str]: list of files within the directory matching the input criteria.
    """
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        all_files += [os.path.join(dirpath, file) for file in filenames]

    if filetype:
        if filetype.lower() != "all":
            all_files = [file for file in all_files if file.endswith(filetype)]

    if contains:
        all_files = [file for file in all_files if contains in file]
        
    if not_contains:
        all_files = [file for file in all_files if not_contains not in file]
    
    return all_files, atmosphere_fps, ocean_fps

def run_PCA(variable_array, num_pcs=None):
    """
    Runs Principal Component Analysis (PCA) on the given variable array.

    Args:
        variable_array (array-like): The input array containing the variables.
        num_pcs (int, optional): The number of principal components to keep. 
            If not specified, all components will be kept.

    Returns:
        tuple: A tuple containing the fitted PCA model and the transformed array.

    """
    if not num_pcs:
        pca = PCA()
    else:
        pca = PCA(n_components=num_pcs)
        
    # if np.isnan(variable_array).any():
    #     return None, None
    
    # TODO: just drop the rows rather than not do the PCA
    pca = pca.fit(variable_array)
    pca_array = pca.transform(variable_array)
    return pca, pca_array

def generate_atmosphere_pcas(atmosphere_fps: list, save_dir: str):
    """
    Generate principal component analysis (PCA) for atmospheric variables.

    Args:
        atmosphere_fps (list): List of file paths to atmospheric CMIP files.
        save_dir (str): Directory to save the PCA results.

    Returns:
        int: Always returns 0.
    """
    
    # for each variable
    var_names = ['pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly', 'smb_anomaly', 'ts_anomaly']
    # var_names = ['mrro_anomaly']
    for var in var_names:
        print(var)
        variable_array = np.zeros([len(atmosphere_fps), 106, 761*761])
        
        # loop through each atmospheric CMIP file
        for i, fp in enumerate(atmosphere_fps):
            
            # get the variable you need (rather than the entire dataset)
            data_flattened, dataset = get_xarray_variable(fp, var)
            
            # store it in the total array
            variable_array[i, :, :] = data_flattened     
            
        
        # deal with np.nans (ask about later)
        variable_array = np.nan_to_num(variable_array)   
        
        # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
        variable_array = variable_array.reshape(len(atmosphere_fps)*len(dataset.time), 761*761)
        
        # run PCA
        # if np.isnan(variable_array).any():
        #     continue
        pca, pca_array = run_PCA(variable_array, num_pcs=None)
        
        # change back to (num_files, num_timestamps, num_pcs)
        pca_array = pca_array.reshape(len(atmosphere_fps), len(dataset.time), -1)
        
        # get percent explained
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        arg_90 = np.argmax(cum_sum_eigenvalues>0.90)+1
        print(f"Variable {var} has {arg_90} PCs that explain 90% of the variance")
        
        # output pca object
        save_path = f"{save_dir}/pca_{var}_{arg_90}pcs.pkl"
        pkl.dump(pca, open(save_path,"wb"))
        np.save(f"{save_dir}/data/{var}_{arg_90}pcs.npy", pca_array)
        
    return 0
        
def get_xarray_variable(dataset_fp, varname):
    """
    Retrieve a variable from an xarray dataset.

    Parameters:
    - dataset_fp (str): Filepath of the xarray dataset.
    - varname (str): Name of the variable to retrieve.

    Returns:
    - data_flattened (ndarray): Flattened array of the variable values.
    - dataset (xarray.Dataset): Original dataset.

    """
    
    dataset = xr.open_dataset(dataset_fp, decode_times=True)
    
    # try dropping dimensions for atmospheric data
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
        data = dataset[varname].values
    except KeyError:
        return np.nan, np.nan
    data_flattened = data.reshape(len(dataset.time), -1)
    # print(len(dataset.time), data_flattened.shape)
    # d = pd.DataFrame(data_flattened)
    # print(d.head())
    
    return data_flattened, dataset

    
        
def generate_ocean_pcas(ocean_fps: list, save_dir: str):
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
        data_flattened, dataset = get_xarray_variable(fp, varname='thermal_forcing')
        thermal_forcing_array[i, :, :] = data_flattened # store
    for i, fp in enumerate(salinity_fps):
        data_flattened, dataset = get_xarray_variable(fp, varname='salinity')
        salinity_array[i, :, :] = data_flattened # store
    for i, fp in enumerate(tempereature_fps):
        data_flattened, dataset = get_xarray_variable(fp, varname='temperature')
        temperature_array[i, :, :] = data_flattened
    
    # reshape variable_array (num_files, num_timestamps, num_gridpoints) --> (num_files*num_timestamps, num_gridpoints)
    thermal_forcing_array = thermal_forcing_array.reshape(len(thermal_forcing_fps)*len(dataset.time), 761*761)
    salinity_array = salinity_array.reshape(len(salinity_fps)*len(dataset.time), 761*761)
    temperature_array = temperature_array.reshape(len(tempereature_fps)*len(dataset.time), 761*761)
    
    # remove nans
    thermal_forcing_array = thermal_forcing_array[:, ~(np.isnan(thermal_forcing_array).any(axis=0))]
    salinity_array = salinity_array[:, ~(np.isnan(salinity_array).any(axis=0))]
    temperature_array = temperature_array[:, ~(np.isnan(temperature_array).any(axis=0))]
    
    # run PCA
    pca_tf, pca_tf_array = run_PCA(thermal_forcing_array, num_pcs=None)
    pca_sal, pca_sal_array = run_PCA(salinity_array, num_pcs=None)
    pca_temp, pca_temp_array = run_PCA(temperature_array, num_pcs=None)
    
    # get percent explained
    tf_exp_var_pca = pca_tf.explained_variance_ratio_
    tf_cum_sum_eigenvalues = np.cumsum(tf_exp_var_pca)
    tf_arg_90 = np.argmax(tf_cum_sum_eigenvalues>0.90)+1
    save_path = f"{save_dir}/pca_thermal_forcing_{tf_arg_90}pcs.pkl"
    pkl.dump(pca_tf, open(save_path,"wb"))
    np.save(f"{save_dir}/data/thermal_forcing_{tf_arg_90}pcs.npy", pca_tf_array)
    
    sal_exp_var_pca = pca_sal.explained_variance_ratio_
    sal_cum_sum_eigenvalues = np.cumsum(sal_exp_var_pca)
    sal_arg_90 = np.argmax(sal_cum_sum_eigenvalues>0.90)+1
    save_path = f"{save_dir}/pca_salinity_{sal_arg_90}pcs.pkl"
    pkl.dump(pca_sal, open(save_path,"wb"))
    np.save(f"{save_dir}/data/salinity_{sal_arg_90}pcs.npy", pca_sal_array)
    
    temp_exp_var_pca = pca_temp.explained_variance_ratio_
    temp_cum_sum_eigenvalues = np.cumsum(temp_exp_var_pca)
    temp_arg_90 = np.argmax(temp_cum_sum_eigenvalues>0.90)+1
    save_path = f"{save_dir}/pca_temperature_{temp_arg_90}pcs.pkl"
    pkl.dump(pca_temp, open(save_path,"wb"))
    np.save(f"{save_dir}/data/temperature_{temp_arg_90}pcs.npy", pca_temp_array)
    
    return 0

def generate_pcas(atmosphere_fps, ocean_fps, save_dir):
    """
    Generate principal component analysis (PCA) for atmosphere and ocean variables. 
    
    Parameters:
    - atmosphere_fps (list): List of file paths for atmosphere data.
    - ocean_fps (list): List of file paths for ocean data.
    - save_dir (str): Directory to save the generated PCA models and results.
    
    Returns:
    - int: 0 indicating successful execution.
    """
    generate_atmosphere_pcas(atmosphere_fps, save_dir)
    generate_ocean_pcas(ocean_fps, save_dir)
    return 0


