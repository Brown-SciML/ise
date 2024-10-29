# Import necessary modules from the ISE package and handle the case when the package is not found.
try:
    from ise.models.ISEFlow import ISEFlow, DeepEnsemble, NormalizingFlow
    from ise.utils import functions as f
    from ise.evaluation import metrics as m
except ImportError:
    import sys
    sys.path.append('/oscar/users/pvankatw/research/ise/')
    from ise.models.ISEFlow import ISEFlow, DeepEnsemble, NormalizingFlow
    from ise.utils import functions as f
    from ise.evaluation import metrics as m

import pandas as pd
import numpy as np
import time
import json

def get_optimal_temponly_model(ice_sheet, out_dir, iterations=10, with_chars=False):
    """
    Trains and evaluates the ISEFlow model for sea level projection based on ice sheet data.

    Args:
        ice_sheet (str): The ice sheet to model ('AIS' for Antarctic or 'GrIS' for Greenland).
        out_dir (str): Directory where the trained model and results will be saved.
        iterations (int, optional): Number of training iterations with different configurations. Default is 10.
        with_chars (bool, optional): Whether to include ice sheet model characteristics in the input features. Default is False.

    Outputs:
        Saves the trained model, predictions, and evaluation metrics to the specified output directory.
    """
    print('With Characteristics:', with_chars)
    print('Iterations:', iterations)
    print('Ice Sheet:', ice_sheet)

    start_time = time.time()
    
    # REPLACE WITH YOUR PATH HERE
    data_dir = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ice_sheet}/"
    print('Data retrieved from:', data_dir)

    # Load training and validation data
    data = pd.read_csv(f"{data_dir}/train.csv")
    val_data = pd.read_csv(f'{data_dir}/val.csv')

    # Prepare input (X) and output (y) datasets
    X_train, y_train = f.get_X_y(data, dataset_type='sectors', return_format='pandas', with_chars=with_chars)
    X_val, y_val = f.get_X_y(val_data, dataset_type='sectors', return_format='pandas', with_chars=with_chars)

    # Concatenate datasets with scenario information
    train = pd.concat([X_train, y_train, data.Scenario], axis=1)
    train['set'] = 'train'
    val = pd.concat([X_val, y_val, val_data.Scenario], axis=1)
    val['set'] = 'val'
    all_data = pd.concat([train, val], axis=0)

    # Split the data back into training and validation sets
    train = all_data[all_data['set'] == 'train'].drop(columns=['set'])
    val = all_data[all_data['set'] == 'val'].drop(columns=['set'])
    scenarios = val['Scenario']

    # Set column names depending on the ice sheet
    if ice_sheet == 'AIS':
        cols = ['smb_anomaly', 'sector', 'year', 'sle']
    else:
        cols = ['aSMB', 'year', 'sle', 'sector']

    print('Using columns:', cols)
    subset_train, subset_val = train[cols], val[cols]

    # Prepare training and validation inputs and outputs
    X_train = subset_train.drop(columns=['sle']).reset_index(drop=True)
    y_train = subset_train['sle'].reset_index(drop=True).values

    X_val = subset_val.drop(columns=['sle']).reset_index(drop=True)
    y_val = subset_val['sle'].reset_index(drop=True).values
    y_val_unscaled = f.unscale(y_val.reshape(-1, 1), f"{data_dir}/scaler_y.pkl")
    years = X_val.year.values

    # Perform model training and evaluation for the specified number of iterations
    for _ in range(iterations):
        for num_predictors in [3, 5, 7, 10]:
            cur_time = time.time()

            # Initialize the model
            de = DeepEnsemble(num_predictors=num_predictors, forcing_size=X_train.shape[1])
            nf = NormalizingFlow(forcing_size=X_train.shape[1])
            emulator = ISEFlow(de, nf)

            # Randomly choose epochs for normalizing flow and deep ensemble training
            nf_epochs = np.random.choice([10, 25, 50])
            de_epochs = np.random.choice([25, 50, 100])

            # Train the model
            train_time_start = time.time()
            print(f"\n\nTraining model with {num_predictors} predictors, {nf_epochs} NF epochs, and {de_epochs} DE epochs")
            emulator.fit(
                X_train, y_train, X_val, y_val,
                early_stopping=True, patience=10, delta=1e-5,
                nf_epochs=nf_epochs, de_epochs=de_epochs,
                early_stopping_path=f"{ice_sheet}_onlysmb_checkpoint.pt"
            )
            train_time_end = time.time()
            total_train_time = (train_time_end - train_time_start) / 60.0

            # Define the model description and save the model
            model_description = f"npred{num_predictors}_nf{nf_epochs}_de{de_epochs}_withchars{with_chars}"
            export_dir = f"{out_dir}/withchars{with_chars}/{model_description}_{cur_time}/"
            emulator.save(export_dir, input_features=cols)

            # Make predictions and save results
            predictions, uncertainties = emulator.predict(X_val, output_scaler=f"{data_dir}/scaler_y.pkl")
            results = pd.DataFrame(dict(year=years, pred=predictions.squeeze(), true=y_val_unscaled.squeeze(), scenarios=scenarios.values.squeeze()))
            results.to_csv(f"{export_dir}/nn_predictions.csv", index=False)

            # Calculate evaluation metrics
            mae = m.mean_absolute_error(y_val_unscaled, predictions)
            mse = m.mean_squared_error(y_val_unscaled, predictions)
            ks_d, ks_p = m.kolmogorov_smirnov(
                results.loc[results.scenarios == 'rcp2.6'].pred,
                results.loc[results.scenarios == 'rcp8.5'].pred
            )
            t_t, t_p = m.t_test(
                results.loc[results.scenarios == 'rcp2.6'].pred,
                results.loc[results.scenarios == 'rcp8.5'].pred
            )

            print(f"MAE: {mae}, MSE: {mse}, KS_D: {ks_d}, KS_P: {ks_p}, Training Time: {total_train_time}")

            # Save metrics
            metrics = dict(model=f"{model_description}_{cur_time}", MAE=mae, MSE=mse, KS_D=ks_d, KS_P=ks_p, Training_Time=total_train_time)
            with open(f'{out_dir}/withchars{with_chars}/metrics.jsonl', 'a') as file:
                json.dump(metrics, file)
                file.write('\n')

if __name__ == '__main__':
    import sys

    # Check if arguments are passed and use defaults otherwise
    if len(sys.argv) == 4:
        ice_sheet = sys.argv[1]
        iterations = int(sys.argv[2])
        with_chars = sys.argv[3].lower() in ('true', 'with_chars')
    else:
        print("No arguments or incorrect arguments passed. Using default values.")
        ice_sheet = 'AIS'
        iterations = 10
        with_chars = False

    # Call the main function to start the model training process
    get_optimal_temponly_model(
        ice_sheet,
        f'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/models/isolated_variables/SMB_only/{ice_sheet}/',
        iterations=iterations,
        with_chars=with_chars
    )
