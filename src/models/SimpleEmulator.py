import torch
from torch import nn

class Emulator(torch.nn.Module):
    def __init__(self, input_layer_size):
        super(Emulator, self).__init__()
        self.input_layer_size = input_layer_size
        self.layer = torch.nn.Linear(self.input_layer_size, 128)
        self.layer2 = torch.nn.Linear(128, 32)
        self.layer3 = torch.nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)     
        return x