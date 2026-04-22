
from ise.models.density_estimators.normalizing_flow import NormalizingFlow
from ise.models.predictors.lstm import LSTM
from ise.utils import functions as f
from ise.utils.functions import to_tensor
from ise.models.ISEFlow import ISEFlow
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import argparse
import wandb
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

def build_model(trial,):
    hidden = trial.suggest_int("lstm_hidden_size", 64, 512, step=32)
    layers = trial.suggest_int("lstm_num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-5, 1e-2,)
    wd = trial.suggest_float("wd", 1e-7, 1e-4,)

    model = LSTM(
        input_size = 75 + z_dim,
        lstm_num_layers=layers,
        lstm_hidden_size=hidden,
        lr=lr,
        wd=wd,
        
    )
    
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    if hasattr(model, 'dropout') and model.dropout is not None:
        model.dropout.p = dropout
    return model

def objective_factory(X_train, y_train, X_val, y_val):
    
    def objective(trial):
        set_seed(seed + trial.number)
        output_dir = f"optuna_ckpts_z{z_dim}_{seed}"
        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"trial_{trial.number}.pt")

        seq_len = trial.suggest_int("sequence_length", 1, 10)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        
        model = build_model(trial, )

        max_epochs = trial.suggest_int("max_epochs", 60, 300, step=20)
        step_epochs = 50
        patience = trial.suggest_int("patience", 10, 50, step=10)
        
        best_seen = np.inf
        for curr_cap in range(step_epochs, max_epochs + 1, step_epochs):
            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                sequence_length=seq_len,
                batch_size=batch_size,
                epochs=curr_cap,
                save_checkpoints=True,
                checkpoint_path=ckpt_path,
                early_stopping=True,
                patience=patience,
                verbose=False,
            )

            with torch.no_grad():
                val_preds = model.predict(X_val, sequence_length=seq_len, batch_size=batch_size).detach().cpu().numpy()
                val_loss = np.mean((y_val.cpu().numpy() - val_preds) ** 2)
                best_seen = min(best_seen, val_loss)
                
                trial.report(best_seen, step=curr_cap)

                if trial.should_prune():
                    raise optuna.TrialPruned()
        return best_seen
    return objective

def main(args):
    
    data = get_data(args.data_dir)
    
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

    sampler = TPESampler(seed=seed, multivariate=True, group=True)
    pruner  = MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=f"single-lstm-latent{z_dim}-{seed}",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    
    # Optimize
    objective = objective_factory(X_train, y_train, X_val, y_val,)
    study.optimize(objective, n_trials=50,)

    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"Value (val MSE): {best.value:.6f}")
    for k, v in sorted(best.params.items()):
        print(f"{k}: {v}")
    

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
    parser = argparse.ArgumentParser(description='Train NF')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the training data directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)

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


# Z=1
# === Best Trial ===
# Value (val MSE): 0.222235
# batch_size: 256
# dropout: 0.0
# lr: 0.0017707864947483318
# lstm_hidden_size: 352
# lstm_num_layers: 1
# max_epochs: 160
# patience: 30
# sequence_length: 5
# wd: 2.914152683267362e-05

# Z=10
# === Best Trial ===
# Value (val MSE): 0.230750
# batch_size: 256
# dropout: 0.0
# lr: 0.0017179603667261244
# lstm_hidden_size: 256
# lstm_num_layers: 1
# max_epochs: 160
# patience: 10
# sequence_length: 10
# wd: 6.0579354733990324e-06
