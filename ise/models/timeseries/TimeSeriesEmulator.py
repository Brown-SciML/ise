import torch
from torch import nn
from ise.models.training.dataclasses import PyTorchDataset, TSDataset
from torch.utils.data import DataLoader
import numpy as np
np.random.seed(10)
import pandas as pd


class TimeSeriesEmulator(torch.nn.Module):
    def __init__(self, architecture,):
        super().__init__()
        self.model_name = 'TimeSeriesEmulator'
        self.input_layer_size = architecture['input_layer_size']
        # self.num_linear_layers = architecture['num_linear_layers']
        # self.nodes = architecture['nodes']
        self.num_rnn_layers = architecture['num_rnn_layers']
        self.num_rnn_hidden = architecture['num_rnn_hidden']
        self.time_series = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determine whether on GPU or not

        if not all([self.num_rnn_layers, self.num_rnn_hidden, ]):
            raise AttributeError('Model architecture argument missing. Requires: [num_rnn_layers, num_rnn_hidden, ].')

        self.rnn = nn.LSTM(
            input_size=self.input_layer_size,
            hidden_size=self.num_rnn_hidden,
            batch_first=True,
            num_layers=self.num_rnn_layers,
            # dropout=0.3,
        )
        
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(in_features=self.num_rnn_hidden, out_features=32)
        self.linear_out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_rnn_layers, batch_size, self.num_rnn_hidden).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_rnn_layers, batch_size, self.num_rnn_hidden).requires_grad_().to(self.device)
        

        _, (hn, _) = self.rnn(x, (h0, c0))
        x = self.linear1(hn[0])
        x = self.relu(x)
        x = self.linear_out(x)
        
        # TODO: Make adjustable number of linear layers and nodes
        # for i in range(self.num_linear_layers):
        #     if i == 0:
        #         x = self.relu(nn.Linear(self.hidden, self.nodes[i]))(x))
        #     elif i == self.num_linear_layers - 1:
        #         x = nn.Linear(self.nodes[i - 1], self.nodes[i]))(x)
        #     else:
        #         x = self.relu(nn.Linear(self.nodes[i - 1], self.nodes[i]))(x))
        return x
    
    def predict(self, x):
        self.eval()
        if isinstance(x, np.ndarray):
            dataset = TSDataset(X=torch.from_numpy(x).float(), y=None, sequence_length=5)
        elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.Tensor):
            dataset = TSDataset(X=x.float(), y=None, sequence_length=5)
        elif isinstance(x, pd.DataFrame):
            dataset = TSDataset(X=torch.from_numpy(np.array(x, dtype=np.float64)).float(), y=None, sequence_length=5)
        else:
            raise ValueError(f'Input x must be of type [np.ndarray, torch.FloatTensor], received {type(x)}')

        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        preds = torch.tensor([]).to(self.device)
        for X_test_batch in loader:
            X_test_batch = X_test_batch.to(self.device)
            test_pred = self(X_test_batch)
            preds = torch.cat((preds, test_pred), 0)
        
        if self.device.type == 'cuda':
            preds = preds.squeeze().cpu().detach().numpy()
        else:
            preds = preds.squeeze().detach().numpy()
        
        return preds