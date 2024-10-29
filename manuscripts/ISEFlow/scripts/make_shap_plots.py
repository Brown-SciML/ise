import pandas as pd
from ise.utils import functions as f
import shap
import matplotlib.pyplot as plt

# Define which ice sheet is being used (AIS for Antarctic Ice Sheet, GrIS for Greenland Ice Sheet)
ICE_SHEET = 'GrIS'

# Column remapping for better readability in SHAP plots
COLUMN_REMAPPING = {
    'numerics': 'Numerics',
    'initial-year': 'Initialization Year',
    'stress-balance': 'Stress Balance Model',
    'resolution': 'Model Resolution',
    'init-method': 'Initialization Method',
    'melt': 'Melt in partially floating cells',
    'ice-front': 'Ice Front Migration Model',
    'open-melt-param': 'Open Basal Melt Parameterization',
    'standard-melt-param': 'Standard Melt Parameterization',
    'Ocean forcing': 'Ocean Forcing Type',
    'Ocean sensitivity': 'Ocean Sensitivity to Basal Melt',
    'Ice shelf fracture': 'Ice Shelf Fracture',
    'mrro_anomaly': 'Runoff Flux',
    'ts_anomaly': 'Surface Air Temperature',
    'smb_anomaly': 'Surface Mass Balance',
    'pr_anomaly': 'Precipitation Flux',
    'temperature': 'Ocean Temperature',
    'evspsbl_anomaly': 'Evaporation Flux',
    'salinity': 'Ocean Salinity',
    'thermal_forcing': 'Ocean Thermal Forcing',
    'year': 'Year',
    'sector': 'Sector',
    'aST': 'Surface Air Temperature',
    'aSMB': 'Surface Mass Balance',
    'basin_runoff': 'Cumulative Basin Runoff',
    'initialization': "Initialization Method",
    'res-max': "Resolution Maximum",
    'res-min': "Resolution Minimum",
    'bed': "Bedrock Topography Model",
    'ghf': "Geothermal Heat Flux",
    'velocity': "Velocity Model",
    'surface-thickness': "Surface/Thickness Model",
    'ice-flow': "Ice Flow Model",
    'initial_smb': "Initial SMB",
}

# Define file paths for SHAP values and sample rows based on ice sheet selection
if ICE_SHEET == 'AIS':
    shap_values_path = r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/explainability/SHAP/SLE_target/finals/AIS/shap_values_50_1724226150.1467326.csv"
else:
    shap_values_path = r'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/explainability/SHAP/SLE_target/finals/GrIS/shap_values_100_1723898221.4189007.csv'

# Load SHAP values and sample rows data
sample_rows_path = shap_values_path.replace('shap_values', 'sample_rows')

shap_values = pd.read_csv(shap_values_path).drop(columns='run_time')
sample_rows = pd.read_csv(sample_rows_path)
sample_rows = sample_rows.drop(columns=[x for x in sample_rows.columns if 'lag' in x])

# Load validation data to match feature columns
dir_ = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ICE_SHEET}/"
X_val_df, y_val_df = f.get_X_y(pd.read_csv(f"{dir_}/val.csv"), 'sectors', return_format='pandas')

# Map column names using the predefined remapping
feature_names = [COLUMN_REMAPPING[x] if x in COLUMN_REMAPPING.keys() else x for x in shap_values.columns]

# Plot and save SHAP summary plot (bar chart)
shap.summary_plot(shap_values.values, feature_names=feature_names, plot_type="bar")
plt.savefig(f'shap_raw_{ICE_SHEET}.png')
plt.savefig(f'shap_raw_{ICE_SHEET}.svg')
plt.close('all')

# Function to aggregate SHAP values by categorical columns with lagging variables
def aggregate_categorical_columns(dataframe):
    """
    Aggregates columns with lagging variables for SHAP values by summing them into their base categories.

    Args:
        dataframe (pd.DataFrame): SHAP values dataframe with lagged variables.
    
    Returns:
        pd.DataFrame: Aggregated dataframe.
    """
    base_variables = ['numerics', 'initial-year', 'stress-balance', 'resolution', 'init-method', 'melt', 'ice-front', 
                      'open-melt-param', 'standard-melt-param', 'Ocean forcing', 'Ocean sensitivity', 'Ice shelf fracture', 
                      'initialization', 'res-max', 'res-min', 'bed', 'ghf', 'velocity', 'surface-thickness', 'ice-flow', 
                      'initial_smb']
    
    aggregated_data = pd.DataFrame()

    # Copy non-lagged columns as is
    for name in dataframe.columns:
        if name.split('.')[0] not in base_variables:
            aggregated_data[name] = dataframe[name]
    
    # Aggregate lagged variables for each base variable
    for base_var in base_variables:
        related_columns = [col for col in dataframe.columns if col.startswith(base_var)]
        if related_columns:
            aggregated_data[base_var] = dataframe[related_columns].sum(axis=1)
    
    return aggregated_data

# Function to format column names by replacing underscores with dashes except the last one
def format_columns(column_strings):
    """
    Replaces underscores in column names with dashes except for the last one in each column name.

    Args:
        column_strings (list of str): Column names.
    
    Returns:
        list of str: Modified column names.
    """
    modified_strings = []
    for s in column_strings:
        underscore_count = s.count('_')
        if underscore_count > 1:
            parts = s.split('_')
            modified_string = '-'.join(parts[:-1]) + '_' + parts[-1]
        else:
            modified_string = s
        modified_strings.append(modified_string)
    
    return modified_strings

# Apply formatting to column names
shap_values.columns = format_columns(shap_values.columns)
shap_values = shap_values.rename(columns={'initial_year': 'initial-year'})

# Aggregate SHAP values by categorical columns
shap_agg = aggregate_categorical_columns(shap_values)

# Plot and save aggregated SHAP values
feature_names = [COLUMN_REMAPPING[x] if x in COLUMN_REMAPPING.keys() else x for x in shap_agg.columns]
shap.summary_plot(shap_agg.values, feature_names=feature_names, plot_type="bar")
plt.savefig(f'shap_summed_by_group_{ICE_SHEET}.png')
plt.savefig(f'shap_summed_by_group_{ICE_SHEET}.svg')
plt.close('all')

# Define forcing columns for each ice sheet
if ICE_SHEET == 'AIS':
    forcing_columns = ['mrro_anomaly', 'ts_anomaly', 'smb_anomaly', 'pr_anomaly', 'temperature',
                       'evspsbl_anomaly', 'salinity', 'thermal_forcing', 'year', 'sector']
else:
    forcing_columns = ['aST', 'aSMB', 'thermal_forcing', 'basin_runoff', 'year', 'sector']

# Drop forcing-related columns and plot SHAP values for model characteristics only
shap_chars = shap_agg.drop(columns=forcing_columns + [x for x in shap_agg.columns if '_' in x])
feature_names = [COLUMN_REMAPPING[x] for x in shap_chars.columns]
shap.summary_plot(shap_chars.values, feature_names=feature_names, plot_type="bar")
plt.savefig(f'shap_chars_{ICE_SHEET}.png')
plt.savefig(f'shap_chars_{ICE_SHEET}.svg')
plt.close('all')

# Scenario-based SHAP plots for different ice sheets
if ICE_SHEET == 'AIS':
    shap_values = pd.read_csv(r'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/explainability/SHAP/Scenario_target/AIS_shap_values_1724177548.840991.csv')
else:
    shap_values = pd.read_csv(r'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/explainability/SHAP/Scenario_target/GrIS_shap_values_1724178272.421653.csv')

# Plot scenario-based SHAP values
feature_names = [COLUMN_REMAPPING[x] if x in COLUMN_REMAPPING.keys() else x for x in shap_values.columns]
shap.summary_plot(shap_values.values, feature_names=feature_names, plot_type="bar")
plt.savefig(f'shap_scenario_{ICE_SHEET}.png')
plt.savefig(f'shap_scenario_{ICE_SHEET}.svg')
plt.close('all')

# Print summary of SHAP values for temperature anomaly (or surface air temperature) and others
mean_shap_values = shap_values.abs().mean()
print(f'Mean Temp SHAP Value ({ICE_SHEET}):', mean_shap_values['ts_anomaly' if ICE_SHEET == 'AIS' else 'aST'])
print(f'Sum of other SHAP Values ({ICE_SHEET}):', mean_shap_values.drop('ts_anomaly' if ICE_SHEET == 'AIS' else 'aST').sum())

stop = ''
