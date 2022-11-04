import torch
from torch import nn

class FC12_N1024(torch.nn.Module):
    def __init__(self, input_layer_size):
        super(FC12_N1024, self).__init__()
        self.input_layer_size = input_layer_size
        self.layer = torch.nn.Linear(self.input_layer_size, 1024)
        self.layer2 = torch.nn.Linear(1024, 900)
        self.layer3 = torch.nn.Linear(900, 512)
        self.layer4 = torch.nn.Linear(512, 256)
        self.layer5 = torch.nn.Linear(256, 128)
        self.layer6 = torch.nn.Linear(128, 64)
        self.layer7 = torch.nn.Linear(64, 32)
        self.layer8 = torch.nn.Linear(32, 16)
        self.layer9 = torch.nn.Linear(16, 8)
        self.layer10 = torch.nn.Linear(8, 4)
        self.layer11 = torch.nn.Linear(4, 2)
        self.layer12 = torch.nn.Linear(2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.relu(self.layer8(x))
        x = self.relu(self.layer9(x))
        x = self.relu(self.layer10(x))
        x = self.relu(self.layer11(x))
        x = self.layer12(x)
        return x