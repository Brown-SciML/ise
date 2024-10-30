try:
    from ise.models.ISEFlow import ISEFlow, DeepEnsemble, NormalizingFlow
    from ise.utils import functions as f
    from ise.evaluation import metrics as m
except:
    import sys
    sys.path.append('/users/pvankatw/research/ise/')
    from ise.models.ISEFlow import ISEFlow, DeepEnsemble, NormalizingFlow
    from ise.utils import functions as f
    from ise.evaluation import metrics as m
import pandas as pd
import numpy as np
import time
import json


def get_best_nn(data_directory, export_directory, iterations=10, with_chars=True):
    X_train, y_train = f.get_X_y(pd.read_csv(f"{data_directory}/train.csv"), 'sectors', return_format='numpy', with_chars=with_chars)
    _, scenarios = f.get_X_y(pd.read_csv(f"{data_directory}/val.csv"), 'scenario', return_format='pandas', with_chars=with_chars)
    X_val_df, _ = f.get_X_y(pd.read_csv(f"{data_directory}/val.csv"), 'sectors', return_format='pandas', with_chars=with_chars)
    X_val, y_val = f.get_X_y(pd.read_csv(f"{data_directory}/val.csv"), 'sectors', return_format='numpy', with_chars=with_chars)
    y_val = f.unscale(y_val.reshape(-1,1), f"{data_directory}/scaler_y.pkl")
    ice_sheet = "AIS" if 'AIS' in data_directory else "GIS"
    print('Cols:', X_val_df.columns)
    
    for _ in range(iterations):
        for num_predictors in [10, 7, 5]:
            
            X_train, y_train = f.get_X_y(pd.read_csv(f"{data_directory}/train.csv"), 'sectors', return_format='numpy', with_chars=with_chars)
            _, scenarios = f.get_X_y(pd.read_csv(f"{data_directory}/val.csv"), 'scenario', return_format='pandas', with_chars=with_chars)
            X_val_df, _ = f.get_X_y(pd.read_csv(f"{data_directory}/val.csv"), 'sectors', return_format='pandas', with_chars=with_chars)
            X_val, y_val = f.get_X_y(pd.read_csv(f"{data_directory}/val.csv"), 'sectors', return_format='numpy', with_chars=with_chars)
            cur_time = time.time()
            de = DeepEnsemble(num_ensemble_members=num_predictors, input_size=X_train.shape[1], )
            nf = NormalizingFlow(input_size=X_train.shape[1])
            emulator = ISEFlow(de, nf)

            nf_epochs = 100
            de_epochs = 100
            train_time_start = time.time()
            print('\n\nTraining model with ', num_predictors, 'predictors,', nf_epochs, 'NF epochs, and', de_epochs, 'DE epochs')
            emulator.fit(
                X_train, y_train, X_val=X_val, y_val=y_val, 
                save_checkpoints=True, checkpoint_path=f"checkpoint_{ice_sheet}",
                early_stopping=True, patience=20, 
                nf_epochs=nf_epochs, de_epochs=de_epochs,
            )
            train_time_end = time.time()
            total_train_time = (train_time_end - train_time_start) / 60.0
            
            model_description = f"npred{num_predictors}_nf{nf_epochs}_de{de_epochs}"
            export_dir = f"{export_directory}/{model_description}_{cur_time}/"
            emulator.save(export_dir)

            predictions, uncertainties = emulator.predict(X_val, output_scaler=f"{data_directory}/scaler_y.pkl")
            y_val = f.unscale(y_val.reshape(-1,1), f"{data_directory}/scaler_y.pkl")
            
            print('Exported to ', f"{export_dir}/nn_predictions.csv")
            results = pd.DataFrame(dict(year=X_val_df.year.values, pred=predictions.squeeze(), true=y_val.squeeze(), scenarios=scenarios.values.squeeze()))
            results.to_csv(f"{export_dir}/nn_predictions.csv", index=False)
            
            mae = m.mean_absolute_error(y_val, predictions)
            mse = m.mean_squared_error(y_val, predictions)
            kld = m.kl_divergence(predictions, y_val)
            jsd = m.js_divergence(predictions, y_val)
            ks_d, ks_p = m.kolmogorov_smirnov(results.loc[results.scenarios == 'rcp2.6'].pred, results.loc[results.scenarios == 'rcp8.5'].pred)
            t_t, t_p = m.t_test(results.loc[results.scenarios == 'rcp2.6'].pred, results.loc[results.scenarios == 'rcp8.5'].pred)
            print(f"MAE: {mae}, MSE: {mse}, KLD: {kld}, JSD: {jsd}, KS_D: {ks_d}, KS_P: {ks_p}, Training Time: {total_train_time}")
            metrics = dict(model=f"{model_description}_{cur_time}", MAE=mae, MSE=mse, KS_D=ks_d, KS_P=ks_p, Training_Time=total_train_time)
            # Append results to a JSON file
            with open(f'{export_directory}/metrics.jsonl', 'a') as file:
                json.dump(metrics, file)
                file.write('\n')
                
if __name__ == '__main__':    
    ICE_SHEET = 'GrIS'
    WITH_CHARS = True
    ITERATIONS = 1
    DATA_DIRECTORY = f'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ICE_SHEET}/'
    EXPORT_DIRECTORY = f'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/models/all_variables/{"with_characteristics" if WITH_CHARS else "without_characteristics"}/{ICE_SHEET}/'
    get_best_nn(DATA_DIRECTORY, EXPORT_DIRECTORY, iterations=ITERATIONS, with_chars=WITH_CHARS)
    
    
    
    