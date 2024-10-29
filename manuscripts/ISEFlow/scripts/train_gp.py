import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
import warnings

# Custom imports from the ISEFlow library
try:
    from ise.models.gp import GP
    from ise.utils import functions as f
    from ise.evaluation import metrics as m
except:
    sys.path.append('/users/pvankatw/research/ise/')
    from ise.models.gp import GP
    from ise.utils import functions as f
    from ise.evaluation import metrics as m

# Ignore warnings during model training
warnings.filterwarnings("ignore")


def train_gp(ice_sheet, temp_only, smb_only):
    """
    Train a Gaussian Process model on ice sheet data and make predictions.

    Args:
        ice_sheet (str): The type of ice sheet being modeled (e.g., 'AIS' or 'GrIS').
        temp_only (bool): Whether to train the model with only temperature-related features.
        smb_only (bool): Whether to train the model with only Surface Mass Balance (SMB) features.

    Returns:
        None. Saves the results and predictions to CSV files.
    """
    start_time = time.time()
    dir_ = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ice_sheet}/"
    print(f'Data retrieved from: {dir_}')
    print(f'Ice sheet: {ice_sheet}, Temperature only: {temp_only}, SMB Only: {smb_only}')

    # Load training, validation, and test datasets
    data = pd.read_csv(f"{dir_}/train.csv")
    val_data = pd.read_csv(f'{dir_}/val.csv')
    test_data = pd.read_csv(f'{dir_}/test.csv')

    # Extract features and target values
    spatial_unit = 'region'
    X_train, y_train, train_scenarios = f.get_X_y(data, dataset_type=f'{spatial_unit}s', return_format='pandas')
    X_val, y_val, val_scenarios = f.get_X_y(val_data, dataset_type=f'{spatial_unit}s', return_format='pandas')
    X_test, y_test, test_scenarios = f.get_X_y(test_data, dataset_type=f'{spatial_unit}s', return_format='pandas')

    # Combine datasets for processing
    train = pd.concat([X_train, y_train, train_scenarios], axis=1)
    train['set'] = 'train'
    val = pd.concat([X_val, y_val, val_scenarios], axis=1)
    val['set'] = 'val'
    test = pd.concat([X_test, y_test, test_scenarios], axis=1)
    test['set'] = 'test'
    all_data = pd.concat([train, val, test], axis=0)

    # Remove duplicates and missing values
    all_data = all_data.drop_duplicates(subset=all_data.columns.difference(['set', 'Scenario']))
    all_data = all_data.dropna()

    # Split the data back into train, validation, and test sets
    train = all_data[all_data['set'] == 'train'].drop(columns=['set'])
    val = all_data[all_data['set'] == 'val'].drop(columns=['set'])
    test = all_data[all_data['set'] == 'test'].drop(columns=['set'])

    # Handle scenarios and additional columns
    train_scenarios = train['Scenario']
    val_scenarios = val['Scenario']
    test_scenarios = test['Scenario']

    # For Antarctic Ice Sheet (AIS), include ice shelf collapse indicator
    if ice_sheet == 'AIS':
        train['ice_shelf_collapse'] = train['Ice shelf fracture_True'] == train['Ice shelf fracture_True'].max()
        val['ice_shelf_collapse'] = val['Ice shelf fracture_True'] == val['Ice shelf fracture_True'].max()
        test['ice_shelf_collapse'] = test['Ice shelf fracture_True'] == test['Ice shelf fracture_True'].max()

    # Select appropriate columns based on the type of model being trained (temp_only, smb_only, or all variables)
    ocean_sensitivity_columns = [x for x in train.columns if 'Ocean sensitivity' in x]
    if ice_sheet == 'AIS':
        if temp_only:
            cols = ['ts_anomaly', spatial_unit, 'ice_shelf_collapse', 'year', 'sle'] + ocean_sensitivity_columns
        elif smb_only:
            cols = ['smb_anomaly', spatial_unit, 'ice_shelf_collapse', 'year', 'sle'] + ocean_sensitivity_columns
        else:
            cols = ['pr_anomaly', 'evspsbl_anomaly', 'ice_shelf_collapse', 'mrro_anomaly', 'smb_anomaly',
                    'ts_anomaly', 'thermal_forcing', 'salinity', 'temperature', 'year', spatial_unit, 'sle'] + ocean_sensitivity_columns
    else:
        if temp_only:
            cols = ['aST', 'year', spatial_unit, 'sle'] + ocean_sensitivity_columns
        elif smb_only:
            cols = ['aSMB', spatial_unit, 'year', 'sle'] + ocean_sensitivity_columns
        else:
            cols = ['aSMB', 'aST', 'thermal_forcing', 'basin_runoff', spatial_unit, 'year', 'sle'] + ocean_sensitivity_columns

    print(f'Columns used for training: {cols}')

    # Subset the data based on selected columns
    subset_train, subset_val, subset_test = train[cols], val[cols], test[cols]

    # Split the data into features (X) and target (y)
    X_train = subset_train.drop(columns=['sle']).reset_index(drop=True)
    y_train = subset_train['sle'].reset_index(drop=True)
    X_val = subset_val.drop(columns=['sle']).reset_index(drop=True)
    y_val = subset_val['sle'].reset_index(drop=True)
    X_test = subset_test.drop(columns=['sle']).reset_index(drop=True)
    y_test = subset_test['sle'].reset_index(drop=True)

    # Unique years in the training data
    all_years = X_train['year'].unique()

    inference_time = 0
    for i in range(86):  # Training for each year
        print(f"Training on year {2015 + i}")
        # Subset the training, validation, and test sets by year
        X_train_subset = X_train[X_train['year'] == all_years[i]].drop(columns=['year'])
        y_train_subset = y_train[X_train['year'] == all_years[i]]
        X_val_subset = X_val[X_val['year'] == all_years[i]].drop(columns=['year'])
        y_val_subset = y_val[X_val['year'] == all_years[i]]
        X_test_subset = X_test[X_test['year'] == all_years[i]].drop(columns=['year'])
        y_test_subset = y_test[X_test['year'] == all_years[i]]

        # Convert subsets to NumPy arrays for model training
        X_train_subset, y_train_subset = X_train_subset.to_numpy(), y_train_subset.to_numpy()
        X_val_subset = X_val_subset.to_numpy()
        X_test_subset = X_test_subset.to_numpy()

        # Define and train the Gaussian Process model
        kernel = ConstantKernel(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1)) + WhiteKernel(noise_level=5, noise_level_bounds=(1e-10, 1e+1))
        model = GP(kernel=kernel)
        model.fit(X_train_subset, y_train_subset)

        # Make predictions and compute upper/lower bounds
        inference_time_start = time.time()
        preds, sd = model.predict(X_train_subset, return_std=True)
        inference_time += (time.time() - inference_time_start) / X_train_subset.shape[0]

        # Store predictions, upper and lower bounds for train, validation, and test sets
        def store_predictions(df, preds, sd, target_df, scaler_path, column='pred'):
            df[column] = f.unscale(preds.reshape(-1, 1), scaler_path).squeeze()
            df['upper_bound'] = f.unscale((preds + sd).reshape(-1, 1), scaler_path).squeeze()
            df['lower_bound'] = f.unscale((preds - sd).reshape(-1, 1), scaler_path).squeeze()

        store_predictions(train.loc[(train['year'] == all_years[i])], preds, sd, y_train_subset, f'{dir_}/scaler_y.pkl')

        preds, sd = model.predict(X_val_subset, return_std=True)
        store_predictions(val.loc[(val['year'] == all_years[i])], preds, sd, y_val_subset, f'{dir_}/scaler_y.pkl')

        preds, sd = model.predict(X_test_subset, return_std=True)
        store_predictions(test.loc[(test['year'] == all_years[i])], preds, sd, y_test_subset, f'{dir_}/scaler_y.pkl')

    # Evaluation and summary statistics
    print(f"Time taken: {time.time() - start_time} seconds")
    print(f"Inference time per 86-year Projection: {inference_time}")
    print(f"MSE on validation set: {np.mean((val['pred'] - val['sle'])**2)}")

    # Statistical differences between scenarios
    print('\nStatistical differences between RCP2.6 and RCP8.5 scenarios:')
    t_values, t_p = m.t_test(val['pred'].loc[val.Scenario == 'rcp8.5'], val['pred'].loc[val.Scenario == 'rcp2.6'])
    ks_values, ks_p = m.kolmogorov_smirnov(val['pred'].loc[val.Scenario == 'rcp8.5'], val['pred'].loc[val.Scenario == 'rcp2.6'])
    print(f'T-Test P Value: {t_p}, KS Test P Value: {ks_p}')

    # Export predictions to CSV
    cur_time = time.time()
    print(f'Exported to: {ice_sheet}_temponly{temp_only}_smbonly{smb_only}_gp_predictions_{cur_time}.csv')
    train.to_csv(f"train_{ice_sheet}_temponly{temp_only}_smbonly{smb_only}_gp_predictions_{cur_time}.csv", index=False)
    val.to_csv(f"val_{ice_sheet}_temponly{temp_only}_smbonly{smb_only}_gp_predictions_{cur_time}.csv", index=False)
    test.to_csv(f"test_{ice_sheet}_temponly{temp_only}_smbonly{smb_only}_gp_predictions_{cur_time}.csv", index=False)


if __name__ == '__main__':
    # Handle command-line arguments for the script
    if len(sys.argv) == 3:
        ice_sheet = sys.argv[1]
        run_type = sys.argv[2]

        # Determine whether to run with temp_only, smb_only, or all variables
        if run_type in ('all_vars', 'temp_only', 'smb_only'):
            temp_only = run_type == 'temp_only'
            smb_only = run_type == 'smb_only'
        else:
            print('Invalid run type. Exiting.')
            exit()
    else:
        print("No valid arguments provided. Using default values.")
        ice_sheet = 'GrIS'
        temp_only = True
        smb_only = False

    train_gp(ice_sheet, temp_only=temp_only, smb_only=smb_only)
