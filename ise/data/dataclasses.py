import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

    def __init__(self, X, y, sequence_length=5, projection_length=86):
        super().__init__()

        if isinstance(projection_length, tuple):
            if len(projection_length) == 1:
                projection_length = projection_length[0]
            else:
                raise ValueError("Projection length must be a single integer or a tuple of two integers.")
        if X.shape[0] < projection_length:
            warnings.warn(
                f"Full projections of {projection_length} timesteps are not present in the dataset. This may lead to unexpected behavior."
            )
        self.X = self._to_tensor(X)
        self.y = self._to_tensor(y)
        self.sequence_length = sequence_length
        self.xdim = len(X.shape)

        if self.xdim == 3:  # batched by projection
            self.num_projections, self.num_timesteps, self.num_features = X.shape
        elif self.xdim == 2:  # unbatched (rows of projections*timestamps)
            self.projections_and_timesteps, self.features = X.shape
            self.num_timesteps = projection_length
            self.num_projections = self.projections_and_timesteps // self.num_timesteps
        # self.num_sequences = self.timesteps - sequence_length + 1

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
            x = torch.tensor(x.values)
        elif isinstance(x, np.ndarray):
            x = torch.tensor(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Data must be a pandas dataframe, numpy array, or PyTorch tensor")
        return x.float()

    def __len__(self):
        if self.xdim == 2:
            return self.X.shape[0]
        else:
            return self.X.shape[0] * self.X.shape[1]

    def __getitem__(self, i):
        """
        Returns the i-th item in the dataset.

        Args:
            i (int): Index of the item to retrieve.

        Returns:
            If `y` is None, returns the input sequence at index `i` as a PyTorch tensor.
            Otherwise, returns a tuple containing the input sequence at index `i` and the corresponding target value.
        """
        # Calculate projection index and timestep index
        projection_index = i // self.num_timesteps
        time_step_index = i % self.num_timesteps

        # Initialize a sequence with zeros for padding
        sequence = torch.zeros((self.sequence_length, self.features))

        # Calculate start and end points for the data to copy from the original dataset
        start_point = max(0, time_step_index - self.sequence_length + 1)
        end_point = time_step_index + 1
        length_of_data = end_point - start_point

        # Copy the data from the dataset to the end of the sequence to preserve recent data at the end
        if self.xdim == 3:
            sequence[-length_of_data:] = self.X[projection_index, start_point:end_point]
        elif self.xdim == 2:
            sequence[-length_of_data:] = self.X[
                projection_index * self.num_timesteps
                + start_point : projection_index * self.num_timesteps
                + end_point
            ]

        if self.y is None:
            return sequence

        return sequence, self.y[i]


class PyTorchDataset(Dataset):
    """
    A PyTorch dataset for general data loading.

    Args:
        X (pandas.DataFrame, numpy.ndarray, or torch.Tensor): The input data.
        y (pandas.DataFrame, numpy.ndarray, or torch.Tensor): The target data.

    Methods:
        __getitem__(index): Returns the item at the given index.
        __len__(): Returns the length of the dataset.
    """

    def __init__(self, X, y):
        self.X_data = X
        self.y_data = y

    def __getitem__(self, index):
        """
        Returns the item at the given index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            If `y` is None, returns the input data at index `index`.
            Otherwise, returns a tuple containing the input data at index `index` and the corresponding target value.
        """
        if self.y_data is None:
            return self.X_data[index]
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.X_data)


class TSDataset(Dataset):
    """
    A PyTorch dataset for time series data.

    Args:
        X (pandas.DataFrame, numpy.ndarray, or torch.Tensor): The input data.
        y (pandas.DataFrame, numpy.ndarray, or torch.Tensor): The target data.
        sequence_length (int): The length of the input sequence.

    Attributes:
        X (torch.Tensor): The input data as a PyTorch tensor.
        y (torch.Tensor): The target data as a PyTorch tensor.
        sequence_length (int): The length of the input sequence.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(i): Returns the i-th item in the dataset.
    """

    def __init__(self, X, y, sequence_length=5):
        super().__init__()
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, i):
        """
        Returns the i-th item in the dataset.

        Args:
            i (int): Index of the item to retrieve.

        Returns:
            If `y` is None, returns the input sequence at index `i` as a PyTorch tensor.
            Otherwise, returns a tuple containing the input sequence at index `i` and the corresponding target value.
        """
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
    

class ScenarioDataset(Dataset):
    def __init__(self, features, labels, ):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
