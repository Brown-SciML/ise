
from ise.models.density_estimators.normalizing_flow import NormalizingFlow
from ise.models.predictors.lstm import LSTM
from ise.utils import functions as f
from ise.utils.functions import to_tensor
from ise.models.ISEFlow import ISEFlow
import argparse
# import wandb
import pandas as pd
import os
import numpy as np
import torch

seed = np.random.randint(1, 10000)
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 1
print(f'z_dim: {z_dim}')

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def sample_loguniform(a, b, size=None, rng=None):
    assert a > 0 and b > a
    rng = rng or np.random.default_rng()
    return np.exp(rng.uniform(np.log(a), np.log(b), size=size))

def randomize_parameters():
    params = {
        "batch_size": int(np.random.choice([64, 128, 256,], p=[1/3, 1/3, 1/3])),
        "dropout": np.random.choice([0.2, 0.3, 0.4], p=[1/3, 1/3, 1/3]),
        "layers": np.random.choice([1, 2, 3], p=[1/3, 1/3, 1/3,]),
        "hidden_size": int(np.round(np.random.normal(loc=350, scale=128))),
        "sequence_length": np.random.choice([5, 10,], p=[0.75, 0.25]),
        "lr": sample_loguniform(5e-4, 9e-3),
        "wd": sample_loguniform(1e-5, 9e-5),
    }
    params['lr'] = params['lr'] * (params['batch_size'] / 128) ** 0.5 
    return params 


def main(args, params):
    
    data = get_data(args.data_dir)
    # wandb.init(
    #     project="iseflow_v1.1.0",  # Your project name
    #     name=f"lstm_{params['hidden_size']}_{params['batch_size']}_{params['lr']}_{params['wd']}",
    #     config={                       # Log hyperparameters
    #         "model": "lstm",
    #     "model_path": args.model_path,
    #     "batch_size": params["batch_size"],
    #     "dropout": params["dropout"],
    #     "layers": params["layers"],
    #     "hidden_size": params["hidden_size"],
    #     "sequence_length": params["sequence_length"],
    #     "lr": params["lr"],
    #     "wd": params["wd"],
    #     "ice_sheet": "GrIS",
    #         },
    #     notes=f"Model: 'lstm', Model path: {args.model_path}, Batch size: {params['batch_size']}, Dropout: {params['dropout']}, Layers: {params['layers']}, Hidden size: {params['hidden_size']}, Sequence length: {params['sequence_length']}, Learning rate: {params['lr']}, Weight decay: {params['wd']}.",  # Additional notes
    #     resume="allow",
    #     id=None,
    #     dir=r"/oscar/scratch/pvankatw/wandb/"
    # )
    
    flow = NormalizingFlow.load(r"/users/pvankatw/research/ise/model/GrIS/nf/nf.pt")
    flow.trained = True

    X_train = to_tensor(data["train"]["X"]).to(device)
    y_train = to_tensor(data["train"]["y"]).to(device)
    X_val = to_tensor(data["val"]["X"]).to(device)
    y_val = to_tensor(data["val"]["y"]).to(device)

    z_train = flow.get_latent(X_train, latent_dim=z_dim).detach().to(device)
    X_train = torch.cat((X_train, z_train), axis=1)

    z_val = flow.get_latent(X_val, latent_dim=z_dim).detach().to(device)
    X_val = torch.cat((X_val, z_val), axis=1)
    
    print(X_train.shape)

    model = LSTM(
        input_size = 90 + z_dim,
        lstm_num_layers=params["layers"],
        lstm_hidden_size=params["hidden_size"],
        lr=params["lr"],
        wd=params["wd"],
        dropout=params["dropout"],
    )
    
    print(model)
    print(params)
    
    model.fit(
        X_train, y_train,
        epochs=1000,
        sequence_length=params["sequence_length"],
        batch_size=params["batch_size"],
        X_val=X_val,
        y_val=y_val,
        save_checkpoints=True,
        checkpoint_path=args.model_path,
        early_stopping=True,
        patience=50,
        # wandb_run=wandb.run,
        

    )
    model_id = args.model_path.split("/")[-1].split(".")[0]
    save_dir = f"model/GrIS/lstm/{model_id}"
    os.makedirs(save_dir, exist_ok=True)
    model.save(f"{save_dir}/lstm.pt")
    
    return 0
    

def get_data(data_dir):
    X_train, y_train = f.get_X_y(pd.read_csv(f"{data_dir}/train.csv"), 'sectors', return_format='numpy',)
    X_val, y_val = f.get_X_y(pd.read_csv(f"{data_dir}/val.csv"), 'sectors', return_format='numpy',)
    X_test, y_test = f.get_X_y(pd.read_csv(f"{data_dir}/test.csv"), 'sectors', return_format='numpy',)
    
    data = {
        "train": {
            "X": X_train,
            "y": y_train
        },
        "val": {
            "X": X_val,
            "y": y_val
        },
        "test": {
            "X": X_test,
            "y": y_test
        }
    }
    return data


def get_args():
    parser = argparse.ArgumentParser(description='Train LSTM')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--model_path', type=str, default="flow_checkpoint.pt", help='Path to the model file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(np.random.randint(1, 10000))
    params = randomize_parameters()
    main(args, params)

# Run program
# uv run python -u main.py \
# --data_dir /users/pvankatw/research/ise/dataset \
# --title "flow_${num_flows}_${hidden_size}_${batch_size}_${lr}_${wd}" \
# --description "Flow, $num_flows flows, hidden size $hidden_size, batch size $batch_size, learning rate $lr, weight decay $wd" \
# --model_path /users/pvankatw/research/ise/model/nf_${flow_id}.pt \
# --num_flows $num_flows \
# --hidden_size $hidden_size \
# --batch_size $batch_size \
# --lr $lr \
# --wd $wd
