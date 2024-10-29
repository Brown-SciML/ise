import json
import pandas as pd
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

# Custom imports from ISEFlow library
sys.path.append('/users/pvankatw/research/ise/')
from ise.evaluation import metrics as m
from ise.utils import functions as f
from ise.models.ISEFlow import DeepEnsemble, ISEFlow, NormalizingFlow
from ise.evaluation.metrics import crps


def get_dataset(ice_sheet='AIS'):
    """
    Load and preprocess the dataset for a given ice sheet.

    Args:
        ice_sheet (str): Ice sheet type ('AIS' for Antarctic Ice Sheet or 'GrIS' for Greenland Ice Sheet).

    Returns:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation targets.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test targets.
    """
    temp_only = True
    smb_only = False
    dir_ = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ice_sheet}/"
    
    # Load train, validation, and test datasets
    data = pd.read_csv(f"{dir_}/train.csv")
    val_data = pd.read_csv(f'{dir_}/val.csv')
    test_data = pd.read_csv(f'{dir_}/test.csv')
    spatial_unit = 'region'
    
    # Extract features and targets
    X_train, y_train, train_scenario = f.get_X_y(data, dataset_type=f'{spatial_unit}s', return_format='pandas')
    X_val, y_val, val_scenario = f.get_X_y(val_data, dataset_type=f'{spatial_unit}s', return_format='pandas')
    X_test, y_test, test_scenario = f.get_X_y(test_data, dataset_type=f'{spatial_unit}s', return_format='pandas')

    # Combine into a single dataset and remove duplicates and NaNs
    train = pd.concat([X_train, y_train, train_scenario], axis=1)
    train['set'] = 'train'
    val = pd.concat([X_val, y_val, val_scenario], axis=1)
    val['set'] = 'val'
    test = pd.concat([X_test, y_test, test_scenario], axis=1)
    test['set'] = 'test'
    all_data = pd.concat([train, val, test], axis=0)
    all_data = all_data.drop_duplicates(subset=all_data.columns.difference(['set', 'Scenario'])).dropna()

    # Split back into train, validation, and test sets
    train = all_data[all_data['set'] == 'train'].drop(columns=['set'])
    val = all_data[all_data['set'] == 'val'].drop(columns=['set'])
    test = all_data[all_data['set'] == 'test'].drop(columns=['set'])

    # Handle Antarctic Ice Sheet (AIS) specific logic
    if ice_sheet == 'AIS':
        for dataset in [train, val, test]:
            dataset['ice_shelf_collapse'] = dataset['Ice shelf fracture_True'] == dataset['Ice shelf fracture_True'].max()

    # Select relevant columns based on `temp_only` flag
    ocean_sensitivity_columns = [x for x in train.columns if 'Ocean sensitivity' in x]
    if ice_sheet == 'AIS':
        cols = ['ts_anomaly', spatial_unit, 'ice_shelf_collapse', 'year', 'sle'] + ocean_sensitivity_columns if temp_only else ['smb_anomaly', spatial_unit, 'ice_shelf_collapse', 'year', 'sle'] + ocean_sensitivity_columns
    else:
        cols = ['aST', 'year', spatial_unit, 'sle'] + ocean_sensitivity_columns if temp_only else ['aSMB', spatial_unit, 'year', 'sle'] + ocean_sensitivity_columns

    # Subset data for selected columns
    subset_train, subset_val, subset_test = train[cols], val[cols], test[cols]

    # Split into features (X) and target (y)
    X_train = subset_train.drop(columns=['sle']).reset_index(drop=True)
    y_train = subset_train['sle'].reset_index(drop=True)
    X_val = subset_val.drop(columns=['sle']).reset_index(drop=True)
    y_val = subset_val['sle'].reset_index(drop=True)
    X_test = subset_test.drop(columns=['sle']).reset_index(drop=True)
    y_test = subset_test['sle'].reset_index(drop=True)

    # Convert 'ice_shelf_collapse' to float for AIS datasets
    if ice_sheet == 'AIS':
        for dataset in [X_train, X_val, X_test]:
            dataset['ice_shelf_collapse'] = dataset['ice_shelf_collapse'].astype(float)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# Load Emulandice prediction data for AIS and GrIS
emulandice_temponly_ais = pd.read_csv("/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/Emulandice/regions/Temp_only/test_AIS_temponlyTrue_smbonlyFalse_gp_predictions_1729644962.7696323.csv")
emulandice_temponly_gris = pd.read_csv("/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/Emulandice/regions/Temp_only/test_GrIS_temponlyTrue_smbonlyFalse_gp_predictions_1729643774.799597.csv")

# Get datasets for AIS and GrIS
X_train_AIS, y_train_AIS, X_val_AIS, y_val_AIS, X_test_AIS, y_test_AIS = get_dataset(ice_sheet='AIS')
X_train_GrIS, y_train_GrIS, X_val_GrIS, y_val_GrIS, X_test_GrIS, y_test_GrIS = get_dataset(ice_sheet='GrIS')

# Directories for AIS and GrIS data
AIS_data_dir = '/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/AIS/'
GrIS_data_dir = '/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/GrIS/'

# Uncomment this section to train and save models
out_dir = '/users/pvankatw/research/ise/supplemental/uq/after_scenario_fix/'
# Train and save ISEFlow models for AIS and GrIS (temp-only version)
# iseflowAIS_temponly = ISEFlow(de, nf)
# iseflowAIS_temponly.fit(X_train_AIS.to_numpy(), y_train_AIS.to_numpy(), ...)
# iseflowAIS_temponly.save(f"{out_dir}/AIS_region_temponly/")

# iseflowGrIS_temponly = ISEFlow(de, nf)
# iseflowGrIS_temponly.fit(X_train_GrIS.to_numpy(), y_train_GrIS.to_numpy(), ...)
# iseflowGrIS_temponly.save(f"{out_dir}/GrIS_region_temponly/")

# Load saved models for AIS and GrIS
iseflowAIS_temponly = ISEFlow.load(f"{out_dir}/AIS_region_temponly/")
iseflowGrIS_temponly = ISEFlow.load(f"{out_dir}/GrIS_region_temponly/")

# Predict using the loaded models and process predictions
smooth = False
iseflowAIS_temponly_preds, iseflowAIS_temponly_uq = iseflowAIS_temponly.predict(X_test_AIS, output_scaler=f"{AIS_data_dir}/scaler_y.pkl", smooth_projection=smooth)
iseflowGrIS_temponly_preds, iseflowGrIS_temponly_uq = iseflowGrIS_temponly.predict(X_test_GrIS, output_scaler=f"{GrIS_data_dir}/scaler_y.pkl", smooth_projection=smooth)

# Unscale the true values for comparison
y_val_AIS = f.unscale(y_val_AIS.values.reshape(-1,1), f"{AIS_data_dir}/scaler_y.pkl")
y_val_GrIS = f.unscale(y_val_GrIS.values.reshape(-1,1), f"{GrIS_data_dir}/scaler_y.pkl")

# Save predictions to CSV (uncomment to save predictions)
# pd.DataFrame(dict(...)).to_csv(f"{out_dir}/AIS_region_temponly/nn_predictions.csv", index=False)
# pd.DataFrame(dict(...)).to_csv(f"{out_dir}/GrIS_region_temponly/nn_predictions.csv", index=False)

# Load previously saved predictions
iseflowAIS_temponly = pd.read_csv(f"{out_dir}/AIS_region_temponly/nn_predictions.csv")
iseflowGrIS_temponly = pd.read_csv(f"{out_dir}/GrIS_region_temponly/nn_predictions.csv")

# Unscale the true values
iseflowAIS_temponly['true'] = f.unscale(iseflowAIS_temponly['true'].values.reshape(-1,1), f"{AIS_data_dir}/scaler_y.pkl")
iseflowGrIS_temponly['true'] = f.unscale(iseflowGrIS_temponly['true'].values.reshape(-1,1), f"{GrIS_data_dir}/scaler_y.pkl")

# Calculate prediction bounds and coverage
for df in [iseflowAIS_temponly, iseflowGrIS_temponly]:
    df['lower_bound'] = df['pred'] - 1.96 * df['uq_total']
    df['upper_bound'] = df['pred'] + 1.96 * df['uq_total']
    df['within_bounds'] = (df['true'] >= df['lower_bound']) & (df['true'] <= df['upper_bound'])

# Calculate the proportion of true values within bounds
proportion_within_bounds_AIS = iseflowAIS_temponly['within_bounds'].mean()
proportion_within_bounds_GrIS = iseflowGrIS_temponly['within_bounds'].mean()

# Calculate MSE
iseflow_AIS_mse = np.mean((iseflowAIS_temponly['true'] - iseflowAIS_temponly['pred']) ** 2)
iseflow_GrIS_mse = np.mean((iseflowGrIS_temponly['true'] - iseflowGrIS_temponly['pred']) ** 2)

# Function to plot projections for AIS and GrIS
def plot_projections(iseflow_data, emulandice_data, ice_sheet, save_dir, num_plots):
    """
    Plot projections for ISEFlow and Emulandice models with uncertainty bands.

    Args:
        iseflow_data (pd.DataFrame): ISEFlow model predictions.
        emulandice_data (pd.DataFrame): Emulandice model predictions.
        ice_sheet (str): Ice sheet type ('AIS' or 'GrIS').
        save_dir (str): Directory to save the plot images.
        num_plots (int): Number of projections to plot.
    """
    years_per_projection = 86
    i = 0
    while i <= num_plots:
        projection_starts = iseflow_data.loc[iseflow_data['year'] == iseflow_data.year.values[0]].index.values
        start = projection_starts[i]
        end = start + years_per_projection

        df_projection = iseflow_data.iloc[start:end]
        emulandice_projection = emulandice_data.iloc[start:end]

        # Plot ISEFlow and Emulandice projections
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.plot(df_projection['year'], df_projection['true'], label='True', color='red', linewidth=2)
        ax1.plot(df_projection['year'], df_projection['pred'], label='Predicted', color='blue', linewidth=2)
        ax1.fill_between(df_projection['year'], df_projection['pred'] - 1.96 * df_projection['uq_epistemic'],
                         df_projection['pred'] + 1.96 * df_projection['uq_epistemic'], color='blue', alpha=0.5)
        ax1.fill_between(df_projection['year'], df_projection['pred'] - 1.96 * df_projection['uq_total'],
                         df_projection['pred'] + 1.96 * df_projection['uq_total'], color='blue', alpha=0.3)
        ax1.set_title(f'{ice_sheet} ISEFlow Projection {i}')
        ax1.legend()

        ax2.plot(emulandice_projection['year'], emulandice_projection['pred'], label='Emulandice Predicted', color='green', linewidth=2)
        ax2.plot(df_projection['year'], df_projection['true'], label='True', color='red', linewidth=2)
        ax2.fill_between(emulandice_projection['year'], emulandice_projection['pred'] - 1.96 * np.abs(emulandice_projection.lower_bound - emulandice_projection.pred),
                         emulandice_projection['pred'] + 1.96 * np.abs(emulandice_projection.lower_bound - emulandice_projection.pred), color='green', alpha=0.3)
        ax2.set_title(f'{ice_sheet} GP Projection {i}')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{save_dir}/projection_{i}.png")
        i += 1
        plt.show()
        plt.close('all')

# Plot AIS and GrIS projections (comment/uncomment as needed)
plot_projections(iseflowAIS_temponly, emulandice_temponly_ais, 'AIS', f'{out_dir}/AIS_region_temponly', 20)
plot_projections(iseflowGrIS_temponly, emulandice_temponly_gris, 'GrIS', f'{out_dir}/GrIS_region_temponly', 20)

# CRPS calculation
i_ais_crps = np.median(crps(iseflowAIS_temponly['true'], iseflowAIS_temponly['pred'], iseflowAIS_temponly['uq_total']))
i_gris_crps = np.median(crps(iseflowGrIS_temponly['true'], iseflowGrIS_temponly['pred'], iseflowGrIS_temponly['uq_total']))
e_ais_crps = np.median(crps(emulandice_temponly_ais['sle'], emulandice_temponly_ais['pred'], np.abs(emulandice_temponly_ais['lower_bound'] - emulandice_temponly_ais['pred'])))
e_gris_crps = np.median(crps(emulandice_temponly_gris['sle'], emulandice_temponly_gris['pred'], np.abs(emulandice_temponly_gris['lower_bound'] - emulandice_temponly_gris['pred'])))

print(f"AIS ISEFlow CRPS: {i_ais_crps}")
print(f"GrIS ISEFlow CRPS: {i_gris_crps}")
print(f"AIS Emulandice CRPS: {e_ais_crps}")
print(f"GrIS Emulandice CRPS: {e_gris_crps}")
