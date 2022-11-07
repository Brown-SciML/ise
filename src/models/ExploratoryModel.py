import torch
from torch import nn

class ExploratoryModel(torch.nn.Module):
    def __init__(self, input_layer_size, num_linear_layers, nodes):
        super(ExploratoryModel, self).__init__()
        self.model_name = 'ExploratoryModel'
        self.input_layer_size = input_layer_size
        self.num_linear_layers = num_linear_layers
        self.nodes = nodes
        
        if len(nodes) != num_linear_layers:
            raise AttributeError(f'Length of nodes argument must be equal to num_linear_layers, received {num_linear_layers} != {len(nodes)}')
        
        if nodes[-1] != 1:
            raise ValueError(f'Last node must be equal to 1, received {nodes[-1]}')
        
        model = nn.Sequential()
        for i in range(num_linear_layers):
            if  i == 0:
                model.append(nn.Linear(self.input_layer_size, nodes[i]))
                model.append(nn.ReLU())
            elif i == num_linear_layers-1:
                model.append(nn.Linear(nodes[i-1], nodes[i]))
            else:
                model.append(nn.Linear(nodes[i-1], nodes[i]))
                model.append(nn.ReLU())
                
            
        self.model = model

    def forward(self, x):
        return self.model(x)