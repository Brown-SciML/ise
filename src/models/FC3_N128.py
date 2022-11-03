import torch
from torch import nn

class FC3_N128(torch.nn.Module):
    def __init__(self, input_layer_size):
        super(FC3_N128, self).__init__()
        self.input_layer_size = input_layer_size
        self.layer = torch.nn.Linear(self.input_layer_size, 128)
        self.layer2 = torch.nn.Linear(128, 64)
        self.layer3 = torch.nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)     
        return x