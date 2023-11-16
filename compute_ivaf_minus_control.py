import xarray as xr
import os
import pandas as pd
import numpy as np
import netCDF4 as nc
import warnings
warnings.simplefilter("ignore") 
# warnings.simplefilter("ignore", category=SerializationWarning)

# # Goelzer et al., 2020 -- https://doi.org/10.5194/tc-14-833-2020
# thif = -(rhow/rhoi)*topg; where (thif<0) thif=0
# af=(lithk-thif)*sftgif*maxmask1*af2; where(af<0) af=0
# ivaf=af.total($x,$y)*dx^2

# thif = ocean_density / ice_density * min(bed_i,0)
# ivaf = (thickness_i + thif) * groundmask_i * mask_i * scalefac_model * (resolution*1000)^2

# thickness_i = lithk
# groundmask_i = sftgrf
# mask_i = sftgif
# scalefac_model = af2

data_directory = r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-GrIS/"
densities_fp = r'/users/pvankatw/research/current/ise/utils/gris_model_densities.csv'
scalefac_fp = r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-GrIS/af2_ISMIP6_GrIS_05000m.nc"

def interpolate_values(data):
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
        
def get_gris_model_densities(zenodo_directory: str, output_path: str=None):
    """Used for getting rhoi and rhow values from the GrIS models outputs.

    Args:
        zenodo_directory (_type_): _description_
        output_path (_type_): _description_

    Returns:
        _type_: _description_
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


def calculate_ivaf_single_file(directory, densities, scalefac_model, ctrl_proj=False):
    
    resolution = 5 #km
    
    path = directory.split('/')
    exp = path[-1]
    model = path[-2]
    group = path[-3]
    
    # MUN_GISM1 is corrupted, skip
    if group == 'MUN' and model == 'GSM1':
        return -1
    
    # exp = 'expd08'
    # model = 'ISSM2'
    # group = 'AWI'
    
    
    # lookup densities from csv
    subset_densities = densities[(densities.group == group) & (densities.model == model)]
    rhoi = subset_densities.rhoi.values[0]
    rhow = subset_densities.rhow.values[0]
    
    # load data
    try: # error with MUN_GSM1 (HDF Error), maybe corrupted? Doesn't work in Jupyter either.
        bed = xr.open_dataset(os.path.join(directory, f'topg_GIS_{group}_{model}_{exp}.nc'))
    except OSError:
        return 0
    thickness = xr.open_dataset(os.path.join(directory, f'lithk_GIS_{group}_{model}_{exp}.nc'))
    mask = xr.open_dataset(os.path.join(directory, f'sftgif_GIS_{group}_{model}_{exp}.nc'))
    ground_mask = xr.open_dataset(os.path.join(directory, f'sftgrf_GIS_{group}_{model}_{exp}.nc'))
    length_time = len(thickness.time)
    
    # fill na values with zero
    
    if np.any(thickness.lithk.isnull()) or np.any(mask.sftgif.isnull()) or np.any(ground_mask.sftgrf.isnull()):
        thickness = thickness.fillna(0)
        mask = mask.fillna(0)
        ground_mask = ground_mask.fillna(0)
        
        # na_values = [np.any(thickness.lithk.isnull()), np.any(mask.sftgif.isnull()), np.any(ground_mask.sftgrf.isnull())]
        # labels = ['thickness', 'mask', 'ground_mask']
        # nas = [labels[i] for i, x in enumerate(na_values) if x]
        # print(f"{group}_{model}_{exp}: Null values found in {nas}, processing unsuccessful.")
        # continue
    
    #! TODO: Ask about this
    if len(set(thickness.y.values)) != len(scalefac_model.y.values):
        bed['x'], bed['y'] = interpolate_values(bed)
        thickness['x'], thickness['y'] = interpolate_values(thickness)
        mask['x'], mask['y'] = interpolate_values(mask)
        ground_mask['x'], ground_mask['y'] = interpolate_values(ground_mask)
        # print(f"{group}_{model}_{exp}: y dimensions do not match scalefac_model, processing unsuccessful.")
        # continue
    
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

def calculate_ivaf_minus_control(data_directory, densities, scalefac_path):
    
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
        
calculate_ivaf_minus_control(data_directory, r"/users/pvankatw/research/current/ise/sectors/utils/gris_model_densities.csv", scalefac_fp)
    



stop = ''


                        