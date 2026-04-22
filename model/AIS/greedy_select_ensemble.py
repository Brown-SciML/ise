import os
from ise.models.ISEFlow import ISEFlow
from ise.models.density_estimators.normalizing_flow import NormalizingFlow
from ise.utils.functions import to_tensor
from ise.utils import functions as f
from ise.models.predictors.lstm import LSTM
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
desired_seq_len = None

def evaluate_ensemble(ensemble, y_val):
    preds = torch.stack([m['preds'].squeeze() for m in ensemble.values()])
    mean_prediction = preds.mean(dim=0)
    return F.mse_loss(mean_prediction, y_val.squeeze())


def main(args):
    # --- Load data ---
    data = get_data(args.data_dir)
    X_train = to_tensor(data["train"]["X"]).to(device)
    y_train = to_tensor(data["train"]["y"]).to(device)
    X_val   = to_tensor(data["val"]["X"]).to(device)
    y_val   = to_tensor(data["val"]["y"]).to(device)
    X_test  = to_tensor(data["test"]["X"]).to(device)
    y_test  = to_tensor(data["test"]["y"]).to(device)
    
    # --- Load flow for latent features ---
    flow = NormalizingFlow.load(r"/users/pvankatw/research/ise/model/AIS/nf/nf.pt")
    flow.trained = True
    z_val = flow.get_latent(X_val, latent_dim=1).detach().to(device)
    X_val = torch.cat((X_val, z_val), axis=1)

    # --- Load models and cache predictions ---
    model_dir = r"/oscar/home/pvankatw/research/ise/model/AIS/lstm"
    all_models = [x for x in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, x))]
    model_paths = [os.path.join(model_dir, m, "lstm.pt") for m in all_models if len(os.listdir(f"{model_dir}/{m}")) > 1]

    models = {}
    for path in model_paths:
        model_id = path.split("/")[-2]
        m = LSTM.load(path)
        models[model_id] = {
            "model": m,
            "best_loss": m.best_loss,
            "preds": m.predict(X_val).detach()
        }

    # --- Start ensemble with the single best model ---
    best_model_id = max(models, key=lambda k: models[k]["best_loss"])
    ensemble = {best_model_id: models[best_model_id]}
    curr_mse = evaluate_ensemble(ensemble, y_val)
    print(f"Start with {best_model_id}: MSE={curr_mse:.4f}")

    max_ensemble_length = 10

    # --- Greedy selection loop ---
    while len(ensemble) < max_ensemble_length:
        best_candidate = None
        best_candidate_mse = curr_mse

        for model_id, model_data in models.items():
            if model_id in ensemble:
                continue
            
            if desired_seq_len is not None and models[model_id]["model"].sequence_length != desired_seq_len:
                continue

            trial_ensemble = ensemble.copy()
            trial_ensemble[model_id] = model_data
            trial_mse = evaluate_ensemble(trial_ensemble, y_val)

            if trial_mse < best_candidate_mse:
                best_candidate = model_id
                best_candidate_mse = trial_mse

        if best_candidate is None:
            print("No further improvement; stopping.")
            break

        # Add best model this round
        ensemble[best_candidate] = models[best_candidate]
        curr_mse = best_candidate_mse
        print(f"Add {best_candidate}: MSE={curr_mse:.4f}")

    print("\nFinal ensemble members:", list(ensemble.keys()))
    print("Final Val MSE:", curr_mse.item())


def get_data(data_dir):
    X_train, y_train = f.get_X_y(pd.read_csv(f"{data_dir}/train.csv"), 'sectors', return_format='numpy',)
    X_val, y_val     = f.get_X_y(pd.read_csv(f"{data_dir}/val.csv"), 'sectors', return_format='numpy',)
    X_test, y_test   = f.get_X_y(pd.read_csv(f"{data_dir}/test.csv"), 'sectors', return_format='numpy',)
    
    return {
        "train": {"X": X_train, "y": y_train},
        "val":   {"X": X_val, "y": y_val},
        "test":  {"X": X_test, "y": y_test}
    }
    



def get_args():
    parser = argparse.ArgumentParser(description='Select an ensemble of LSTM models using greedy selection.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the training data directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)


# Start with lstm_6827: MSE=0.3176
# Add lstm_14993: MSE=0.2339
# Add lstm_26595: MSE=0.2144
# Add lstm_4500: MSE=0.2075
# Add lstm_4271: MSE=0.2043
# Add lstm_8468: MSE=0.2026
# Add lstm_13654: MSE=0.2014
# Add lstm_31659: MSE=0.2003
# Add lstm_4084: MSE=0.1995
# Add lstm_17368: MSE=0.1991

# Final ensemble members: ['lstm_6827', 'lstm_14993', 'lstm_26595', 'lstm_4500', 'lstm_4271', 'lstm_8468', 'lstm_13654', 'lstm_31659', 'lstm_4084', 'lstm_17368']
# Final Val MSE: 0.19906963407993317