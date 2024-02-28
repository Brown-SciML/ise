import torch
import pandas as pd
import numpy as np

class Scaler:
    def __init__(self,):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = self._to_tensor(X)
        self.mean_ = torch.mean(X, dim=0)
        self.scale_ = torch.std(X, dim=0, unbiased=False)
        # self.scale_ = torch.where(self.scale_ == 0, torch.ones_like(self.scale_) * self.eps, self.scale_)  # Avoid division by zero

    def transform(self, X):
        X = self._to_tensor(X)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("This StandardScalerPyTorch instance is not fitted yet.")
        transformed = (X - self.mean_) / self.scale_
        
        # handle NAN (i.e. divide by zero)
        # could also use epsilon value and divide by epsilon instead...
        if torch.isnan(transformed).any():
            transformed = torch.nan_to_num(transformed)
            
        return transformed

    def inverse_transform(self, X):
        X = self._to_tensor(X)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("This StandardScalerPyTorch instance is not fitted yet.")
        return X * self.scale_ + self.mean_

    def save(self, path):
        torch.save({
            'mean_': self.mean_,
            'scale_': self.scale_,
        }, path)
        
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

    @staticmethod
    def load(path):
        checkpoint = torch.load(path)
        scaler = Scaler()
        scaler.mean_ = checkpoint['mean_']
        scaler.scale_ = checkpoint['scale_']
        return scaler
