import torch
from torch import nn


class ExploratoryModel(torch.nn.Module):
    def __init__(self, input_layer_size, architecture,):
        super(ExploratoryModel, self).__init__()
        self.model_name = 'ExploratoryModel'
        self.input_layer_size = input_layer_size
        self.num_linear_layers = architecture['num_linear_layers']
        self.nodes = architecture['nodes']

        if len(self.nodes) != self.num_linear_layers:
            raise AttributeError(
                f'Length of nodes argument must be equal to num_linear_layers, received {self.num_linear_layers} != {len(self.nodes)}')

        if self.nodes[-1] != 1:
            raise ValueError(f'Last node must be equal to 1, received {self.nodes[-1]}')

        model = nn.Sequential()
        for i in range(self.num_linear_layers):
            if i == 0:
                model.append(nn.Linear(self.input_layer_size, self.nodes[i]))
                model.append(nn.ReLU())
            elif i == self.num_linear_layers - 1:
                model.append(nn.Linear(self.nodes[i - 1], self.nodes[i]))
            else:
                model.append(nn.Linear(self.nodes[i - 1], self.nodes[i]))
                model.append(nn.ReLU())

        self.model = model

    def forward(self, x):
        return self.model(x)
