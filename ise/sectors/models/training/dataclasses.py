from torch.utils.data import Dataset
import torch


class PyTorchDataset(Dataset):
    def __init__(self, X, y):
        self.X_data = X
        self.y_data = y

    def __getitem__(self, index):
        if self.y_data is None:
            return self.X_data[index]
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TSDataset(Dataset):
    def __init__(self, X, y, sequence_length=5):
        super().__init__()
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        if self.y is None:
            return x

        return x, self.y[i]
