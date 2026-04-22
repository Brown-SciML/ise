
from ise.utils import functions as f
from ise.models.density_estimators.normalizing_flow import NormalizingFlow
import argparse
import wandb
import pandas as pd
import os


def main(args):
    
    data = get_data(args.data_dir)
    # wandb.init(
    #     project="iseflow_v1.1.0",  # Your project name
    #     name=f"{args.title}",          # Unique name for this run
    #     config={                       # Log hyperparameters
    #         "model": "nf",
    #     "model_path": args.model_path,
    #     "num_flows": args.num_flows,
    #     "hidden_size": args.hidden_size,
    #     "batch_size": args.batch_size,
    #     "lr": args.lr,
    #     "wd": args.wd,
    #     "ice_sheet": "GrIS",
    #         },
    #     notes=f"Model: 'nf', {args.description}",
    #     resume="allow",
    #     id=args.wandb_id if args.wandb_id else None,
    #     dir=r"/oscar/scratch/pvankatw/wandb/"
    # )
    
    flow = NormalizingFlow(
        input_size=90,
        output_size=1,
        num_flow_transforms=args.num_flows,
        flow_hidden_features=args.hidden_size
    )
    flow.fit(
        X=data['train']['X'],
        y=data['train']['y'],
        X_val=data['val']['X'],
        y_val=data['val']['y'],
        epochs=1000,
        batch_size=args.batch_size,
        save_checkpoints=True,
        checkpoint_path=args.model_path,
        early_stopping=True,
        patience=50,
        verbose=True,
        # wandb_run=wandb,
        lr=args.lr,
        wd=args.wd,
    )

    model_id = args.model_path.split("/")[-1].split(".")[0]
    save_dir = f"model/GrIS/nf/{model_id}"
    os.makedirs(save_dir, exist_ok=True)
    flow.save(f"{save_dir}/best_model.pt")
    
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
    parser.add_argument('--title', type=str, required=True, help='WandB title for the run')
    parser.add_argument('--description', type=str, required=True, help='WandB description for the run')
    parser.add_argument('--wandb_id', type=str, default=None, help='WandB ID for resuming runs')
    parser.add_argument('--model_path', type=str, default="flow_checkpoint.pt", help='Path to the model file')
    parser.add_argument('--num_flows', type=int, default=5, help='Number of flow transformations')
    parser.add_argument('--hidden_size', type=int, default=16, help='Hidden layer size for the flow model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay for training')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(f"Training flow with args: {args}")
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