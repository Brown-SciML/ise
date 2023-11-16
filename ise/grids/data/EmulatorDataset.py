

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class EmulatorDataset(Dataset):
    """
    A PyTorch dataset for loading emulator data.

    Args:
        X (pandas.DataFrame, numpy.ndarray, or torch.Tensor): The input data.
        y (pandas.DataFrame, numpy.ndarray, or torch.Tensor): The target data.
        sequence_length (int): The length of the input sequence.

    Attributes:
        X (torch.Tensor): The input data as a PyTorch tensor.
        y (torch.Tensor): The target data as a PyTorch tensor.
        sequence_length (int): The length of the input sequence.

    Methods:
        __to_tensor(x): Converts input data to a PyTorch tensor.
        __len__(): Returns the length of the dataset.
        __getitem__(i): Returns the i-th item in the dataset.
    """
    def __init__(self, X, y, sequence_length=5):
        super().__init__()
        self.X = self._to_tensor(X)
        self.y = self._to_tensor(y)
        self.sequence_length = sequence_length


    def _to_tensor(self, x):
        """
        Converts input data to a PyTorch tensor of type float.

        Args:
            x: Input data to be converted. Must be a pandas dataframe, numpy array, or PyTorch tensor.

        Returns:
            A PyTorch tensor of type float.
        """
        if x is None:
            return None
        if isinstance(x, pd.DataFrame):
            x = x.values
        elif isinstance(x, np.ndarray):
            x = torch.tensor(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Data must be a pandas dataframe, numpy array, or PyTorch tensor")
        return x.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), dim=0)

        if self.y is None:
            return x

        return x, self.y[i]
