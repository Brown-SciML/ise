import xarray as xr
import os
import pandas as pd
import numpy as np
import netCDF4 as nc
import warnings
import os
warnings.simplefilter("ignore") 


class GrISProcessor:
    def __init__(self, forcings_directory, projections_directory, scalefac_path=None, densities_path=None):
        """
        Initializes a GrISProcessor object with the given parameters.

        Args:
            forcings_directory (str): The path to the directory containing the forcing data.
            projections_directory (str): The path to the directory containing the projection data.
            scalefac_path (str): The path to the file containing the scaling factors.
            densities_path (str, optional): The path to the file containing the densities data. Defaults to None.
        """
        self.forcings_directory = forcings_directory
        self.projections_directory = projections_directory
        self.densities_path = densities_path
        self.scalefac_path = scalefac_path
        
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
        if not os.path.exists(f"{self.projections_directory}/VUW/PISM/exp08/ivaf_GIS_VUW_PISM_exp08.nc"):
            _calculate_ivaf_minus_control(self.projections_directory, self.densities_path, self.scalefac_path)
        
        
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
        
        
        
        
    def _calculate_ivaf_minus_control(data_directory: str, densities_fp: str, scalefac_path: str):
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
            densities = pd.read_csv(densities)
        elif isinstance(densities_fp, pd.DataFrame):
            pass
        else:
            raise ValueError("densities argument must be a string or a pandas DataFrame.")
        
        scalefac_model = xr.open_dataset(scalefac_path)
        
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
            calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=True)
        
        # then, for each experiment, calculate ivaf and subtract out control
        for directory in exp_dirs:
            calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=False)
            
            
        return 1
        

    def _calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=False):
        """
        Calculates the ice volume above flotation (IVAF) for a single file in a given directory.
        
        Args:
        - directory (str): The directory path where the file is located.
        - densities (pandas.DataFrame): A DataFrame containing the densities of ice and water for each group and model.
        - scalefac_model (xarray.Dataset): An xarray Dataset containing the scale factor for each model.
        - ctrl_proj (bool): A boolean indicating whether the calculation is for a control projection or not.
        
        Returns:
        - int: 1 if processing is successful, -1 if unsuccessful.
        """
    
        resolution = 5 #km
        
        path = directory.split('/')
        exp = path[-1]
        model = path[-2]
        group = path[-3]
        
        # MUN_GISM1 is corrupted, skip
        if group == 'MUN' and model == 'GSM1':
            return -1
        
        # lookup densities from csv
        subset_densities = densities[(densities.group == group) & (densities.model == model)]
        rhoi = subset_densities.rhoi.values[0]
        rhow = subset_densities.rhow.values[0]
        
        # load data
        bed = xr.open_dataset(os.path.join(directory, f'topg_GIS_{group}_{model}_{exp}.nc'))
        thickness = xr.open_dataset(os.path.join(directory, f'lithk_GIS_{group}_{model}_{exp}.nc'))
        mask = xr.open_dataset(os.path.join(directory, f'sftgif_GIS_{group}_{model}_{exp}.nc'))
        ground_mask = xr.open_dataset(os.path.join(directory, f'sftgrf_GIS_{group}_{model}_{exp}.nc'))
        length_time = len(thickness.time)
        
        # fill na values with zero
        if np.any(thickness.lithk.isnull()) or np.any(mask.sftgif.isnull()) or np.any(ground_mask.sftgrf.isnull()):
            thickness = thickness.fillna(0)
            mask = mask.fillna(0)
            ground_mask = ground_mask.fillna(0)
        
        
        #! TODO: Ask about this
        if len(set(thickness.y.values)) != len(scalefac_model.y.values):
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
        
        ivaf = np.zeros(bed.topg.values.shape)
        for i in range(length_time):
            # bed_values = bed.topg.values[i,:,:] if len(bed.topg.dims) == 3 else bed.topg.values # sometimes time is missing for dims, so just use x,y 
            thif = rhow / rhoi * np.min(bed.topg.values[i,:,:],0)
            masked_output = (thickness.lithk[i, :, :] + thif) * ground_mask.sftgrf[i, :, :] * mask.sftgif[i, :, :]
            ivaf[i, :, :] =  masked_output * scalefac_model.af2.values * (resolution*1000)**2
            
        # subtract out control if for an experment
        ivaf_nc = bed.copy()  # copy file structure and metadata for ivaf file
        if not ctrl_proj:
            # open control dataset
            ivaf_ctrl = xr.open_dataset(os.path.join("/".join(path[:-1]), f'ctrl_proj/ivaf_GIS_{group}_{model}_ctrl_proj.nc'))
            
            # if the time lengths don't match (one goes for 85 years and the other 86) select only time frames that match
            if ivaf_ctrl.time.values.shape[0] > ivaf.shape[0]:
                ivaf_ctrl = ivaf_ctrl.isel(time=slice(0,ivaf.shape[0]))
                ivaf_nc = ivaf_nc.drop_sel(time=ivaf_nc.time.values[:(ivaf.shape[0]-ivaf_ctrl.time.values.shape[0])]) # drop extra time steps
            elif ivaf_ctrl.time.values.shape[0] < ivaf.shape[0]:
                ivaf_nc = ivaf_nc.drop_sel(time=ivaf_nc.time.values[ivaf_ctrl.time.values.shape[0]-ivaf.shape[0]:]) # drop extra time steps
                ivaf = ivaf[0:ivaf_ctrl.time.values.shape[0],:,:]
                
            else:
                pass
                
            ivaf = ivaf_ctrl.ivaf.values - ivaf
        else:
            pass
            
        # save ivaf file
        ivaf_nc['ivaf'] = (('time', 'y', 'x'), ivaf)
        ivaf_nc = ivaf_nc.drop_vars(['topg',])
        ivaf_nc.to_netcdf(os.path.join(directory, f'ivaf_GIS_{group}_{model}_{exp}.nc'))

        print(f"{group}_{model}_{exp}: Processing successful.")
        
        return 1
        
        



def get_gris_model_densities(zenodo_directory: str, output_path: str=None):
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
                    dataset = xr.open_dataset(file_path)

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
        if 'ctrl_proj' in file['filename']:
            continue
        elif 'ILTS' in file['filename']:
            fp = file['filename'].split('_')
            group = 'ILTS_PIK'
            model = fp[-2]  
        else:
            fp = file['filename'].split('_')
            group = fp[-3]
            model = fp[-2]
        densities.append([group, model, file['rhoi'], file['rhow']])

    df = pd.DataFrame(densities, columns=['group', 'model', 'rhoi', 'rhow'])
    df['rhoi'], df['rhow'] = df.rhoi.astype('float'), df.rhow.astype('float')
    df = df.drop_duplicates()
    
    if output_path is not None:
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