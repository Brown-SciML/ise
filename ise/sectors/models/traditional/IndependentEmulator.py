import torch
from torch import nn
from ise.sectors.models.training.dataclasses import PyTorchDataset
from torch.utils.data import DataLoader
import numpy as np

np.random.seed(10)
import pandas as pd
import torch


class IndependentEmulator(torch.nn.Module):
    def __init__(self, architecture, mc_dropout=False, dropout_prob=None):
        super().__init__()
        self.model_name = "IndependentEmulator"
        self.input_layer_size = architecture["input_layer_size"]
        self.num_linear_layers = architecture["num_linear_layers"]
        self.num_nodes_hidden = architecture["num_nodes_hidden"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mc_dropout = mc_dropout

        if not all(
            [
                self.num_linear_layers,
                self.num_nodes_hidden,
            ]
        ):
            raise AttributeError(
                "Model architecture argument missing. Requires: [num_rnn_layers, num_rnn_hidden, ]."
            )

        if mc_dropout and dropout_prob is None:
            raise ValueError("If mc_dropout, dropout_prob cannot be None.")
            
        self.linear_in = nn.Linear(in_features=self.input_layer_size, out_features=self.num_nodes_hidden)
        self.linear_hidden = nn.Linear(in_features=self.num_nodes_hidden, out_features=self.num_nodes_hidden)
        self.relu = nn.ReLU()
        if self.mc_dropout:
            self.dropout = nn.Dropout(p=dropout_prob)
        self.linear1 = nn.Linear(in_features=self.num_nodes_hidden, out_features=32)
        self.linear_out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear_in(x)
        
        if self.num_linear_layers > 1:
            for _ in range(self.num_linear_layers):
                x = self.linear_hidden(x)
        if self.mc_dropout:
            x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        if self.mc_dropout:
            x = self.dropout(x) 
        x = self.linear_out(x)
        
        return x

    def predict(self, x, approx_dist=None, mc_iterations=None, quantile_range=[0.025, 0.975], confidence="95"):
        
        approx_dist = self.mc_dropout if approx_dist is None else approx_dist
        if approx_dist and mc_iterations is None:
            raise ValueError(
                "If the model was trained with MC Dropout, mc_iterations cannot be None."
            )

        self.eval()
        if isinstance(x, np.ndarray):
            dataset = PyTorchDataset(
                X=torch.from_numpy(x).float(), y=None, 
            )
        elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.Tensor):
            dataset = PyTorchDataset(X=x.float(), y=None, )
        elif isinstance(x, pd.DataFrame):
            dataset = PyTorchDataset(
                X=torch.from_numpy(np.array(x, dtype=np.float64)).float(),
                y=None,
               
            )
        else:
            raise ValueError(
                f"Input x must be of type [np.ndarray, torch.FloatTensor], received {type(x)}"
            )

        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        iterations = 1 if not approx_dist else mc_iterations
        out_preds = np.zeros([iterations, len(dataset)])

        for i in range(iterations):
            preds = torch.tensor([]).to(self.device)
            for X_test_batch in loader:
                self.eval()
                self.enable_dropout()
                if approx_dist:
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
