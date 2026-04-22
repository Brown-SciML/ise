from ise.models.ISEFlow import ISEFlow
from ise.utils.functions import to_tensor
from ise.utils import functions as f
import pandas as pd
import argparse
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    
    data = get_data(args.data_dir)
    X_train = to_tensor(data["train"]["X"]).to(device)
    y_train = to_tensor(data["train"]["y"]).to(device)
    X_val   = to_tensor(data["val"]["X"]).to(device)
    y_val   = to_tensor(data["val"]["y"]).to(device)
    X_test  = to_tensor(data["test"]["X"]).to(device)
    y_test  = to_tensor(data["test"]["y"]).to(device)
    
    iseflow = ISEFlow.load(args.model_dir )
    # iseflow = ISEFlow.load(r"/users/pvankatw/research/ise/model/ISEFlow/", )
    
    preds, uq = iseflow.predict(X_test, output_scaler=f"{args.model_dir}/scaler_y.pkl")
    unscaled_y_test = f.unscale_output(y_test.cpu().numpy(), f"{args.model_dir}/scaler_y.pkl")
    
    preds = preds.squeeze()
    y_true = unscaled_y_test.squeeze()

    mse = ((preds - y_true) ** 2).mean()
    print("Test MSE:", mse)
    



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
    parser = argparse.ArgumentParser(description='Test ISEFlow model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--model_dir', type=str, required=True, default=None, help='Path to the saved ISEFlow model directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)