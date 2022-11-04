import torch
from torch import nn

class FC6_N256(torch.nn.Module):
    def __init__(self, input_layer_size):
        super(FC6_N256, self).__init__()
        self.num_fc = 6
        self.num_nodes_max = 256
        self.model_name = 'FC6_N256'
        self.input_layer_size = input_layer_size
        self.layer = torch.nn.Linear(self.input_layer_size, 256)
        self.layer2 = torch.nn.Linear(256, 128)
        self.layer3 = torch.nn.Linear(128, 64)
        self.layer4 = torch.nn.Linear(64, 32)
        self.layer5 = torch.nn.Linear(32, 16)
        self.layer6 = torch.nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return x