import os
import torch
import pandas as pd
import numpy as np
import time
import json

# Import modules from the ISE package, and handle the case when the package is not found.
try:
    from ise.models.ISEFlow import ISEFlow, DeepEnsemble, NormalizingFlow, WeakPredictor
    from ise.utils import functions as f
    from ise.evaluation import metrics as m
except ModuleNotFoundError:
    import sys
    sys.path.append('/users/pvankatw/research/ise/')
    from ise.models.ISEFlow import ISEFlow, DeepEnsemble, NormalizingFlow, WeakPredictor
    from ise.utils import functions as f
    from ise.evaluation import metrics as m

def get_optimal_weakpredictor(ice_sheet, out_dir, iterations=10):
    """
    Trains a WeakPredictor model for sea level projections based on ice sheet data.

    Args:
        ice_sheet (str): The ice sheet to model ('AIS' for Antarctic or 'GrIS' for Greenland).
        out_dir (str): Directory where the trained model and results will be saved.
        iterations (int, optional): Number of iterations to run the model training with different configurations. Default is 10.

    Outputs:
        - Saves the trained model and predictions in the specified output directory.
        - Outputs evaluation metrics (MAE, MSE, KS_D, KS_P) for each model configuration.
        - Appends the metrics for each iteration to a JSONL file.
    """
    
    start_time = time.time()
    data_dir = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ice_sheet}/"
    print('Data retrieved from:', data_dir)

    # Perform model training and evaluation for the specified number of iterations
    for _ in range(iterations):
        
        # Load training and validation data
        X_train, y_train = f.get_X_y(pd.read_csv(f"{data_dir}/train.csv"), 'sectors', return_format='numpy')
        _, scenarios = f.get_X_y(pd.read_csv(f"{data_dir}/val.csv"), 'scenario', return_format='pandas')
        X_val_df, _ = f.get_X_y(pd.read_csv(f"{data_dir}/val.csv"), 'sectors', return_format='pandas')
        years = X_val_df.year.values
        X_val, y_val = f.get_X_y(pd.read_csv(f"{data_dir}/val.csv"), 'sectors', return_format='numpy')

        # Initialize the WeakPredictor model
        lstm_num_layers = 1
        lstm_hidden_size = 512
        emulator = WeakPredictor(
            lstm_num_layers=lstm_num_layers, 
            lstm_hidden_size=lstm_hidden_size, 
            input_size=X_train.shape[1], 
            ice_sheet=ice_sheet, 
            criterion=torch.nn.HuberLoss()
        )

        # Define training parameters
        epochs = 200
        train_time_start = time.time()
        print(f"\n\nTraining model with {lstm_num_layers} num_layers, {lstm_hidden_size} hidden_size, and {epochs} epochs")

        # Train the model
        emulator.fit(
            X_train, y_train, 
            X_val=X_val, y_val=y_val, 
            early_stopping=True, patience=20, delta=1e-5, 
            epochs=epochs, early_stopping_path=f'wp_early_stopping_{ice_sheet}.pt'
        )
        train_time_end = time.time()
        total_train_time = (train_time_end - train_time_start) / 60.0
        
        # Save the model
        cur_time = time.time()
        model_description = f"wp_layers{lstm_num_layers}_hidden{lstm_hidden_size}_epoch{epochs}"
        export_dir = f"{out_dir}/{model_description}_{cur_time}/"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        torch.save(emulator.state_dict(), f"{export_dir}/wp_model.pth")

        # Make predictions on the validation set
        predictions = emulator.predict(X_val)
        predictions = f.unscale(predictions.cpu().detach().numpy(), f"{data_dir}/scaler_y.pkl")
        y_val = f.unscale(y_val.reshape(-1, 1), f"{data_dir}/scaler_y.pkl")

        # Save predictions to CSV
        print('Exported to ', f"{export_dir}/nn_predictions.csv")
        results = pd.DataFrame(dict(year=years, pred=predictions.squeeze(), true=y_val.squeeze(), scenarios=scenarios.values.squeeze()))
        results.to_csv(f"{export_dir}/nn_predictions.csv", index=False)

        # Calculate evaluation metrics
        mae = m.mean_absolute_error(y_val, predictions)
        mse = m.mean_squared_error(y_val, predictions)
        ks_d, ks_p = m.kolmogorov_smirnov(
            results.loc[results.scenarios == 'rcp2.6'].pred, 
            results.loc[results.scenarios == 'rcp8.5'].pred
        )
        t_t, t_p = m.t_test(
            results.loc[results.scenarios == 'rcp2.6'].pred, 
            results.loc[results.scenarios == 'rcp8.5'].pred
        )
        print(f"MAE: {mae}, MSE: {mse}, KS_D: {ks_d}, KS_P: {ks_p}, Training Time: {total_train_time}")

        # Save metrics to JSONL
        metrics = dict(
            model=f"{model_description}_{cur_time}", 
            MAE=mae, MSE=mse, KS_D=ks_d, KS_P=ks_p, Training_Time=total_train_time
        )
        with open(f'{out_dir}/metrics.jsonl', 'a') as file:
            json.dump(metrics, file)
            file.write('\n')


if __name__ == '__main__':
    # Example usage with default arguments for the Greenland Ice Sheet (GrIS)
    get_optimal_weakpredictor(
        'GrIS', 
        f'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/models/WeakPredictors/', 
        iterations=100
    )
    # Uncomment the following line for Antarctic Ice Sheet (AIS) modeling
    # get_optimal_weakpredictor('AIS', f'/users/pvankatw/research/current/supplemental/model_optimization/AIS/weakpredictor/', iterations=100)
