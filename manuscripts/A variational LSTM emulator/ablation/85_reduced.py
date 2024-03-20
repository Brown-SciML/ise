import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ise.models.training.dataclasses import PyTorchDataset
from ise.utils.data import load_ml_data


class YearlyModel(torch.nn.Module):
    def __init__(self, architecture, dropout_prob=0.2):
        super().__init__()
        self.model_name = "YearlyModel"
        self.input_layer_size = architecture["input_layer_size"]
        self.num_nodes = architecture["num_nodes"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_main = nn.Linear(self.input_layer_size, self.num_nodes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear1 = nn.Linear(in_features=self.num_nodes, out_features=32)
        self.linear_out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = self.relu(self.linear_main(x))
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear_out(x)
        return x

    def predict(
        self,
        x,
        mc_iterations=None,
    ):

        self.eval()
        if isinstance(x, np.ndarray):
            dataset = PyTorchDataset(
                torch.from_numpy(x).float(),
                None,
            )
        elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.Tensor):
            dataset = PyTorchDataset(
                x.float(),
                None,
            )
        elif isinstance(x, pd.DataFrame):
            dataset = PyTorchDataset(
                torch.from_numpy(np.array(x, dtype=np.float64)).float(),
                None,
            )
        else:
            raise ValueError(
                f"Input x must be of type [np.ndarray, torch.FloatTensor], received {type(x)}"
            )

        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        iterations = mc_iterations
        out_preds = np.zeros([iterations, len(dataset)])

        for i in range(iterations):
            preds = torch.tensor([]).to(self.device)
            for X_test_batch in loader:
                self.eval()
                self.enable_dropout()

                X_test_batch = X_test_batch.to(self.device)
                test_pred = self(X_test_batch)
                preds = torch.cat((preds, test_pred), 0)

            if self.device.type == "cuda":
                preds = preds.squeeze().cpu().detach().numpy()
            else:
                preds = preds.squeeze().detach().numpy()
            out_preds[i, :] = preds

        if 1 in out_preds.shape:
            out_preds = out_preds.squeeze()

        means = out_preds.mean(axis=0)
        sd = out_preds.std(axis=0)

        return out_preds, means, sd

    def enable_dropout(
        self,
    ):
        # For each layer, if it starts with Dropout, turn it from eval mode to train mode
        for layer in self.modules():
            if layer.__class__.__name__.startswith("Dropout"):
                layer.train()


print("Loading data...")
train_features, train_labels, test_features, test_labels, test_scenarios = load_ml_data(
    data_directory=r"/users/pvankatw/emulator/untracked_folder/ml_data",
)
data_dict = {
    "train_features": train_features,
    "train_labels": train_labels,
    "test_features": test_features,
    "test_labels": test_labels,
}

criterion = nn.MSELoss()
individual_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

year_results = []
for year in tqdm(train_features.year.unique()):

    train_features_year = train_features[train_features.year == year]
    train_labels_year = np.array(train_labels[train_labels.index.isin(train_features_year.index)])
    test_features_year = test_features[test_features.year == year]
    test_labels_year = np.array(test_labels[test_labels.index.isin(test_features_year.index)])

    columns = [c for c in test_features_year.columns if "lag" not in c]
    train_features_year = np.array(train_features_year[columns])
    test_features_year = np.array(test_features_year[columns])

    architecture = {
        "input_layer_size": train_features_year.shape[1],
        "num_nodes": 64,
    }
    model = YearlyModel(architecture=architecture, dropout_prob=0.2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
    )

    train_dataset = PyTorchDataset(
        torch.from_numpy(train_features_year).float(),
        torch.from_numpy(train_labels_year).float().squeeze(),
    )
    test_dataset = PyTorchDataset(
        torch.from_numpy(test_features_year).float(),
        torch.from_numpy(test_labels_year).float().squeeze(),
    )

    # Create dataset and data loaders to be used in training loop
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=256,
    )

    # Loop through epochs
    for epoch in range(1, individual_epochs + 1):
        model.train()
        epoch_start = time.time()

        total_loss = 0
        total_mae = 0

        # for each batch in train_loader
        for X_train_batch, y_train_batch in train_loader:

            # send to gpu if available
            X_train_batch = X_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)

            # set gradients to zero for the batch
            optimizer.zero_grad()

            # get prediction and calculate loss
            pred = model(X_train_batch)
            loss = criterion(pred, y_train_batch.unsqueeze(1))

            # calculate dloss/dx for every parameter x (gradients) and advance optimizer
            loss.backward()
            optimizer.step()

            # add loss to total loss
            total_loss += loss.item()

        # divide total losses by number of batches and save to logs
        avg_mse = total_loss / len(train_loader)

    raw_preds, preds, sd = model.predict(test_features_year, mc_iterations=100)

    year_df = test_features[test_features.year == year].copy()
    year_df["preds"] = preds
    year_df["std"] = sd
    year_results.append(year_df)

year_results = pd.concat(year_results).sort_index()

print("MSE:", np.mean((year_results["preds"] - test_labels) ** 2))
year_results.to_csv("/users/pvankatw/emulator/untracked_folder/baylor_tests/85_full.csv")
