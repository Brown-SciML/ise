import torch
import pandas as pd
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components  # Can be an int or a float between 0 and 1
        self.mean = None
        self.components = None
        self.singular_values = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # Center the data
        X = self._to_tensor(X)
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean

        # Compute low-rank PCA
        U, S, V = torch.pca_lowrank(X_centered, q=301)
        
        # Compute total variance
        total_variance = S.pow(2).sum()
        explained_variance = S.pow(2)
        explained_variance_ratio = explained_variance / total_variance
        
        # Determine the number of components for the desired variance
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)
            # Number of components to explain desired variance
            num_components = torch.searchsorted(cumulative_variance_ratio, self.n_components) + 1
            self.n_components = min(num_components, S.size(0))
        elif isinstance(self.n_components, int):
            self.n_components = min(self.n_components, S.size(0))
        else:
            raise ValueError("n_components must be a float in the range (0, 1) or an integer.")
        
        self.components = V[:, :self.n_components]
        self.singular_values = S[:self.n_components]
        self.explained_variance = explained_variance[:self.n_components]
        self.explained_variance_ratio = explained_variance_ratio[:self.n_components]
        return self

    def transform(self, X):
        X = self._to_tensor(X)
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        X_centered = X - self.mean
        return torch.mm(X_centered, self.components)

    def inverse_transform(self, X,):
        X = self._to_tensor(X)
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        inverse = torch.mm(X, self.components.t()) + self.mean
        return inverse

    def save(self, path):
        torch.save({
            'n_components': self.n_components,
            'mean': self.mean,
            'components': self.components,
            'singular_values': self.singular_values,
            'explained_variance': self.explained_variance,
            'explained_variance_ratio': self.explained_variance_ratio
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
        model = PCA(checkpoint['n_components'])
        model.mean = checkpoint['mean']
        model.components = checkpoint['components']
        model.singular_values = checkpoint['singular_values']
        model.explained_variance = checkpoint['explained_variance']
        model.explained_variance_ratio = checkpoint['explained_variance_ratio']
        return model
