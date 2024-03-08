import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ise.models.timeseries.TimeSeriesEmulator import TimeSeriesEmulator
from ise.models.training.dataclasses import TSDataset
from ise.models.training.Trainer import Trainer
from ise.utils.data import load_ml_data

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
individual_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get all columns associated with the variable
ts_anomaly_cols = [c for c in train_features.columns if "ts_anomaly" in c]
salinity_cols = [c for c in train_features.columns if "salinity" in c]
temperature_cols = [c for c in train_features.columns if "temperature" in c]
columns = ts_anomaly_cols + salinity_cols + temperature_cols  # + ['year', 'sectors']
train_features = np.array(train_features[columns])
test_features = np.array(test_features[columns])

train_dataset = TSDataset(
    torch.from_numpy(train_features).float(),
    torch.from_numpy(np.array(train_labels)).float().squeeze(),
    sequence_length=5,
)
test_dataset = TSDataset(
    torch.from_numpy(test_features).float(),
    torch.from_numpy(np.array(test_labels)).float().squeeze(),
    sequence_length=5,
)

# Create dataset and data loaders to be used in training loop
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=256,
)


architecture = {
    "input_layer_size": train_features.shape[1],
    "num_rnn_layers": 1,
    "num_rnn_hidden": 256,
}
model = TimeSeriesEmulator(architecture=architecture, mc_dropout=True, dropout_prob=0.2).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
)

# Loop through epochs
for epoch in tqdm(range(1, individual_epochs + 1)):
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


raw_preds, preds, sd = model.predict(test_features, mc_iterations=100)


out_df = pd.DataFrame(
    dict(preds=preds, sd=sd),
)
out_df.to_csv("/users/pvankatw/emulator/untracked_folder/baylor_tests/1_reduced.csv")
