from ise.models.ISEFlow import ISEFlow
from ise.utils.functions import to_tensor
from ise.utils import functions as f
import pandas as pd
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    
    data = get_data(args.data_dir)
    X_test  = to_tensor(data["test"]["X"]).to(device)
    y_test  = to_tensor(data["test"]["y"]).to(device)
    
    iseflow = ISEFlow.load(args.model_dir )
    # iseflow = ISEFlow.load(r"/users/pvankatw/research/ise/model/ISEFlow/", )
    
    preds, uq = iseflow.predict(X_test, output_scaler=f"{args.model_dir}/scaler_y.pkl", smoothing_window=0)
    unscaled_y_test = f.unscale_output(y_test.cpu().numpy(), f"{args.model_dir}/scaler_y.pkl")
    

    
    preds = preds.squeeze()
    y_true = unscaled_y_test.squeeze()

    out = pd.DataFrame(data["test"]["X"])
    out['preds_unscaled'] = preds
    out['y_true_unscaled'] = y_true
    out.to_csv(f"{args.model_dir}/test_predictions.csv", index=False)

    mse = ((preds - y_true) ** 2).mean()
    print("Test MSE:", mse)
    
    i = 0
    if args.plot:
        for i in range(10):
            # i = np.random.randint(0, len(preds)//86)
            preds_plot = preds[i*86:(i+1)*86]
            y_true_plot = y_true[i*86:(i+1)*86]
            epistemic = uq['epistemic'][i*86:(i+1)*86].squeeze()
            aleatoric = uq['aleatoric'][i*86:(i+1)*86].squeeze()
            total = uq['total'][i*86:(i+1)*86].squeeze()
            
            plt.figure(figsize=(12, 6))
            plt.plot(preds_plot, label='Predictions')
            plt.plot(y_true_plot, label='True')
            plt.fill_between(np.arange(1, 87), preds_plot - epistemic, preds_plot + epistemic, color='blue', alpha=0.5, label='Emulator Uncertainty (2$sigma$)')
            plt.fill_between(np.arange(1, 87), preds_plot - total, preds_plot + total, color='green', alpha=0.5, label='Data Coverage Uncertainty')
            plt.title('Predictions vs True Values (Sample)')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f'predictions_{i}.png')
            plt.show()
    



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
    parser.add_argument('--plot', action='store_true', default=False, help='Whether to plot the results')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)