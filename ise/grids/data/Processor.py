import xarray as xr
import os
import pandas as pd
import numpy as np
import netCDF4 as nc
import warnings
import os
warnings.simplefilter("ignore") 


class Processor:
    def __init__(self, ice_sheet, forcings_directory, projections_directory, scalefac_path=None, densities_path=None):
        self.forcings_directory = forcings_directory
        self.projections_directory = projections_directory
        self.densities_path = densities_path
        self.scalefac_path = scalefac_path
        self.ice_sheet = ice_sheet.upper()
        self.resolution = 5 if self.ice_sheet.lower() in ('gris', 'gis') else 8
        
    def process_forcings(self):
        if self.forcings_directory is None:
            raise ValueError("Forcings path must be specified")
        
        pass
    
    def process_projections(self, output_directory):
        if self.projections_directory is None:
            raise ValueError("Projections path must be specified")
        
        if output_directory is None:
            raise ValueError("Output directory must be specified")
        
        # if the last ivaf file is missing, assume none of them are and calculate and export all ivaf files
        if self.ice_sheet == "AIS" and not os.path.exists(f"{self.projections_directory}/VUW/PISM/exp08/ivaf_GIS_VUW_PISM_exp08.nc"):
            self._calculate_ivaf_minus_control(self.projections_directory, self.densities_path, self.scalefac_path)
        elif self.ice_sheet in ("GrIS", "GIS") and not os.path.exists(f"{self.projections_directory}/VUW/PISM/exp04/ivaf_AIS_VUW_PISM_exp04.nc"):
            self._calculate_ivaf_minus_control(self.projections_directory, self.densities_path, self.scalefac_path)
        
        
        # get ivaf files in projections directory
        ivaf_files = []
        for root, dirs, files in os.walk(self.projections_directory):
            for file in files:
                if 'ivaf' in file and 'ctrl_proj' not in file:
                    ivaf_files.append(os.path.join(root, file))
        
        # create array of ivaf values of shape (num_files, num_timestamps, y, x) and store in projections_array
        projections_array = np.zeros([len(ivaf_files), 86, 577, 337])
        metadata = []
        for i, file in enumerate(ivaf_files):
            data = xr.open_dataset(file)
            ivaf = data.ivaf.values
            if ivaf.shape[0] != 86:
                ivaf = ivaf[0:86,:,:]
            projections_array[i, :, :, :] = ivaf
            
            # capture metadata for storage
            path = file.split('/')
            exp = path[-2]
            model = path[-3]
            group = path[-4]
            metadata.append([group, model, exp])
            
        # save projections array and metadata
        np.save(f"{output_directory}/projections_array.npy", projections_array)
        metadata_df = pd.DataFrame(metadata, columns=['group', 'model', 'exp'])
        metadata_df.to_csv(f"{output_directory}/projections_metadata.csv", index=False)
            
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
        - int: 1 (dummy value)
        
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
        
        scalefac_model = xr.open_dataset(scalefac_path)
        scalefac_model = np.transpose(scalefac_model.af2.values, (1,0))
        
        if self.ice_sheet == "AIS":
            scalefac_model = scalefac_model[::self.resolution, ::self.resolution]
        
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
        
        # delete later
        for directory in exp_dirs[399:]:
            self._calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=False)
            
        # first calculate ivaf for control projections
        for directory in ctrl_proj_dirs:
            self._calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=True)
        
        # then, for each experiment, calculate ivaf and subtract out control
        for directory in exp_dirs:
            self._calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=False)
            
            
        return 1
        

    def _calculate_ivaf_single_file(self, directory, densities, scalefac_model, ctrl_proj=False):                
        path = directory.split('/')
        exp = path[-1]
        model = path[-2]
        group = path[-3]
        
        # Determine which control to use based on experiment (only applies to AIS) per Nowicki, 2020
        if not ctrl_proj:
            if self.ice_sheet.lower() == 'ais':
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
            
        bed = xr.open_dataset(os.path.join(directory, f'topg_{naming_convention}'), decode_times=False)
        thickness = xr.open_dataset(os.path.join(directory, f'lithk_{naming_convention}'), decode_times=False)
        mask = xr.open_dataset(os.path.join(directory, f'sftgif_{naming_convention}'), decode_times=False)
        ground_mask = xr.open_dataset(os.path.join(directory, f'sftgrf_{naming_convention}'), decode_times=False)
        length_time = len(thickness.time)
        
        
        
        # NCAR_CISM_expD1: Processing successful.
        # Traceback (most recent call last):
        # File "/oscar/home/pvankatw/research/current/main.py", line 20, in <module>
        #     processor.process_projections(output_directory=r"/users/pvankatw/research/current/supplemental/processed_data/")
        # File "/oscar/home/pvankatw/research/current/ise/grids/data/Processor.py", line 35, in process_projections
        #     self._calculate_ivaf_minus_control(self.projections_directory, self.densities_path, self.scalefac_path)
        # File "/oscar/home/pvankatw/research/current/ise/grids/data/Processor.py", line 121, in _calculate_ivaf_minus_control
        #     self._calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=False)
        # File "/oscar/home/pvankatw/research/current/ise/grids/data/Processor.py", line 195, in _calculate_ivaf_single_file
        #     if len(set(thickness.y.values)) != len(scalefac_model):
        # File "/users/pvankatw/anaconda/emulator/lib/python3.10/site-packages/xarray/core/common.py", line 246, in __getattr__
        #     raise AttributeError(
        # AttributeError: 'Dataset' object has no attribute 'y'
        #         #
        
        #! TODO: Ask about this
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
            
        
        bed_data = np.transpose(bed.topg.values, (2,1,0))
        thickness_data = np.transpose(thickness.lithk.values, (2,1,0))
        mask_data = np.transpose(mask.sftgif.values, (2,1,0))
        ground_mask_data = np.transpose(ground_mask.sftgrf.values, (2,1,0))
        
        ivaf = np.zeros(bed_data.shape)
        for i in range(length_time):   
            
            # get data slices for current time         
            thickness_i = thickness_data[:,:,i]
            bed_i = bed_data[:,:,i]
            mask_i = mask_data[:,:,i]
            ground_mask_i = ground_mask_data[:,:,i]
            
            # set data slices to zero where mask = 0 or any value is NaN
            thickness_i[(mask_i == 0) | (np.isnan(mask_i))| (np.isnan(thickness_i))| (np.isnan(ground_mask_i))| (np.isnan(bed_i))] = 0
            bed_i[(mask_i == 0) | (np.isnan(mask_i))| (np.isnan(thickness_i))| (np.isnan(ground_mask_i))| (np.isnan(bed_i))] = 0
            ground_mask_i[(mask_i == 0) | np.isnan(mask_i) | np.isnan(thickness_i)| np.isnan(ground_mask_i)| np.isnan(bed_i)] = 0
            mask_i[(mask_i == 0) | (np.isnan(mask_i))| (np.isnan(thickness_i))| (np.isnan(ground_mask_i))| (np.isnan(bed_i))] = 0
            
            # take min(bed_i, 0)
            bed_i[bed_i > 0] = 0
            
            
            # thif = rhow / rhoi * bed_i
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
                datetimeindex = ivaf_ctrl.indexes['time'].to_datetimeindex()
                ivaf_ctrl['time'] = datetimeindex
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
    
    ice_sheet = "AIS" if "AIS" in file['filename'] else "GrIS"
    
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


