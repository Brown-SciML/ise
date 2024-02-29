

from torch.utils.data import Dataset
import torch
import pandas as pd
import warnings
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
        
        if X.shape[0] < 86:
            warnings.warn("Full projections of 86 timesteps are not present in the dataset. This may lead to unexpected behavior.")
        self.X = self._to_tensor(X)
        self.y = self._to_tensor(y)
        self.sequence_length = sequence_length
        self.xdim = len(X.shape)
        
        if self.xdim == 3: # batched by projection
            self.num_projections, self.num_timesteps, self.num_features = X.shape
        elif self.xdim == 2: # unbatched (rows of projections*timestamps)
            self.projections_and_timesteps, self.features = X.shape
            self.num_timesteps = 86
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
            sequence[-length_of_data:] = self.X[projection_index*self.num_timesteps + start_point:projection_index*self.num_timesteps + end_point]
        
        
        if self.y is None:
            return sequence

        return sequence, self.y[i]
