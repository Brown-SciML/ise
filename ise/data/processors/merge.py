import pandas as pd
import matplotlib.pyplot as plt
import re
import requests
import json


# Open up the JSON with Table 1 from H. Seroussi et al.: ISMIP6 Antarctica projections
# Link: https://tc.copernicus.org/articles/14/3033/2020/tc-14-3033-2020.pdf
resp = requests.get(r'https://raw.githubusercontent.com/pvankatwyk/ise/master/ise/data/datasets/processed_output_files/ismip6_experiments.json')
ismip6_experiments = json.loads(resp.text)


# ismip6_experiments_filepath = r'https://github.com/pvankatwyk/emulator/blob/master/src/data/processed_output_files/ismip6_experiments.json'
# with open(ismip6_experiments_filepath) as experiments:
#     ismip6_experiments = json.load(experiments)


def merge_datasets(processed_forcing_directory, processed_ismip6_directory, export_directory, include_icecollapse=False):
    master, inputs, outputs = combine_datasets(processed_forcing_directory=processed_forcing_directory,
                                               processed_ismip6_directory=processed_ismip6_directory,
                                               include_icecollapse=include_icecollapse,
                                               export=export_directory)
    return master, inputs, outputs

def combine_datasets(processed_forcing_directory, processed_ismip6_directory, include_icecollapse, export=True):
    """Combines the input datasets -- atmospheric forcing, three oceanic forcing (salinity, temperature
    and thermal forcing), and ice sheet collapse forcing with the output dataset generated in 
    H. Seroussi et al.: ISMIP6 Antarctica projections.
    

    Args:
        processed_data_dir (str, optional): Directory of the processed files. Should contain
            atmospheric_forcing, ice_collapse, and three oceanic forcing CSV's. Defaults to export_dir.
        include_icecollapse (boolean, optional): Include the ice collapse parameter
            in generating the dataset. Defaults to with_ice.
        export (str/bool, optional): Directory of exported files. Defaults to export_dir.

    Returns:
        master (pandas.DataFrame): Master dataset (inputs and outputs)
        inputs (pandas.DataFrame): Input dataset (atmospheric, oceanic, ice collapse)
        outputs (pandas.DataFrame): Output dataset (Zenodo Scalers from Seroussi et al.)
    """
    
    # Get the files and if that doesn't work, return a FIleNotFoundError
    try:
        af = pd.read_csv(f"{processed_forcing_directory}/atmospheric_forcing.csv")
        ice = pd.read_csv(f"{processed_forcing_directory}/ice_collapse.csv")
        salinity = pd.read_csv(f"{processed_forcing_directory}/salinity.csv")
        temp = pd.read_csv(f"{processed_forcing_directory}/temperature.csv")
        tf = pd.read_csv(f"{processed_forcing_directory}/thermal_forcing.csv")
        outputs = pd.read_csv(f"{processed_ismip6_directory}/ismip6_outputs.csv")
    except FileNotFoundError:
        raise FileNotFoundError('Files not found, make sure to run all processing functions.')
    
    
    # Merge the oceanic datasets together (thermal forcing, temperature, salinity)
    # and apply the format_aogcms function for formatting strings in AOGCM column
    ocean = salinity
    ocean['aogcm'] = ocean['aogcm'].apply(format_aogcms)  
    af['aogcm'] = af['aogcm'].apply(format_aogcms)
    for data in [temp, tf,]:
        data['aogcm'] = data['aogcm'].apply(format_aogcms)
        ocean = pd.merge(ocean, data, on=['sectors', 'year', 'aogcm', 'regions'], how="outer")
    ocean = ocean.drop_duplicates()
    ocean = ocean[['sectors', 'regions', 'year', 'aogcm', 'salinity', 'temperature', 'thermal_forcing']]
    
    # Apply the same formatting function to atmospheric and ice forcing
    af['aogcm'] = af['aogcm'].apply(format_aogcms)
    ice['aogcm'] = ice['aogcm'].apply(format_aogcms)
    
    # Merge all inputs into one dataframe using an inner join
    inputs = pd.merge(ocean, af, on=['sectors', 'year', 'aogcm', 'regions'], how="inner")
    
    # If indicated, add ice collapse
    if include_icecollapse:
        inputs = pd.merge(inputs, ice, on=['sectors', 'year', 'aogcm', 'regions'], how="inner")
    
    # Map the experiment to attribute function, which takes Table 1 from H. Seroussi et al.: ISMIP6 Antarctica projections
    # and adds columns for other attributes listed in the table...
    outputs['experiment'], outputs['aogcm'], outputs['scenario'], outputs['ocean_forcing'], outputs['ocean_sensitivity'], outputs['ice_shelf_fracture'], outputs['tier'] = zip(*outputs['exp_id'].map(exp_to_attributes))
    
    # Merge inputs and outputs
    master = pd.merge(inputs, outputs, on=['year', 'sectors', 'aogcm'])
    
    if export:
        master.to_csv(f"{export}/master.csv", index=False)
        inputs.to_csv(f"{export}/inputs.csv", index=False)
        outputs.to_csv(f"{export}/outputs.csv", index=False)
    
    return master, inputs, outputs
    
    
    

def exp_to_attributes(x,):
    """Combines Table 1 in H. Seroussi et al.: ISMIP6 Antarctica projections and associates the attributes
    listed in Table 1 with each experiment in the output dataset.

    Args:
        x (str): AOGCM string as it is stored in the 'aogcm' column

    Returns:
        attributes: Returns all new attributes associated with each experiment
    """
    try:
        attributes = ismip6_experiments[x]
        return attributes['Experiment'], attributes['AOGCM'], attributes['Scenario'], attributes['Ocean forcing'], attributes['Ocean sensitivity'], attributes['Ice shelf fracture'], attributes['Tier'] 
    except:
        pass


def format_aogcms(x):
    """Formats AOGCM strings so that joins between datasets work properly. This is necessary due to
    differing file directory names in the original AIS Globus dataset.

    Args:
        x (str): AOGCM string as it is stored in the 'aogcm' column

    Returns:
        x (str): Formatted AOGCM string
    """
    # To homogeonize, get rid of periods (rcp85 vs rcp8.5) and make all dashes underscores
    x = x.lower().replace(".", "").replace("-", "_")
    try:
        try:
            # If it is already in the format "ssp585", do nothing and continue
            correct_format = re.search("(ssp|rcp)\d{2,3}", x).group()
        except AttributeError:
            # If not, find the numeric value (e.g. 85) and change to rcp (_rcp85)
            numeric = re.search("\d{2,3}", x).group()
            x = x[:-2]
            if x.endswith('_'):
                x += f"rcp{numeric}"
            else:
                x += f"_rcp{numeric}"
                
    except AttributeError:
        # if none of the above worked, just skip it
        pass
    
    # Get rid of _1 and include case for ukesm1_0_ll to match other formats
    x = x.replace("_1", "")
    if x == "ukesm1_0_ll":
        x += "_ssp585"
    return x
