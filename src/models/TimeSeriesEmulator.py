import torch
from torch import nn


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

        if not all([self.num_rnn_layers, self.num_rnn_hidden, self.input_layer_size]):
            raise AttributeError('Model architecture argument missing. Requires: [num_rnn_layers, num_rnn_hidden, input_layer_size].')

        self.rnn = nn.LSTM(
            input_size=self.input_layer_size,
            hidden_size=self.num_rnn_hidden,
            batch_first=True,
            num_layers=self.num_rnn_layers
        )

        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(in_features=self.num_rnn_hidden, out_features=32)
        self.linear_out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_rnn_layers, batch_size, self.num_rnn_hidden).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_rnn_layers, batch_size, self.num_rnn_hidden).requires_grad_().to(self.device)
        

        _, (hn, _) = self.rnn(x, (h0, c0))
        out = self.linear1(hn[0])
        out = self.relu(out)
        out = self.linear_out(out)
        return out