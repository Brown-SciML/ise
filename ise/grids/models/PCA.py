import torch
import pandas as pd
import numpy as np

class PCA:
    def __init__(self, n_components):
        
        """
        Principal Component Analysis (PCA) model.

        This class provides a PCA model which can be fit to data.

        Attributes:
            n_components (int or float): The number of components to keep. Can be an int or a float between 0 and 1.
            mean (torch.Tensor): The mean of the input data. Calculated during fit.
            components (torch.Tensor): The principal components. Calculated during fit.
            singular_values (torch.Tensor): The singular values corresponding to each of the principal components. Calculated during fit.
            explained_variance (torch.Tensor): The amount of variance explained by each of the selected components. Calculated during fit.
            explained_variance_ratio (torch.Tensor): Percentage of variance explained by each of the selected components. Calculated during fit.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.singular_values = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        Fit the PCA model to the input data.

        Args:
            X (np.array | pd.DataFrame): Input data, a tensor of shape (n_samples, n_features).

        Returns:
            self (PCModel): The fitted PCA model.

        Raises:
            ValueError: If n_components is not a float in the range (0, 1) or an integer.
        """
        # Center the data
        X = self._to_tensor(X)
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean

        # Compute low-rank PCA
        U, S, V = torch.pca_lowrank(X_centered, q=301,)

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
        """
        Apply dimensionality reduction to the input data using the fitted PCA model.

        Args:
            X (np.array | pd.DataFrame): Input data, a tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Transformed data, a tensor of shape (n_samples, n_components).

        Raises:
            RuntimeError: If the PCA model has not been fitted yet.
        """
        X = self._to_tensor(X)
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        X_centered = X - self.mean
        return torch.mm(X_centered, self.components)

    def inverse_transform(self, X):
        """
        Apply inverse dimensionality reduction to the input data using the fitted PCA model.

        Args:
            X (np.array | pd.DataFrame): Transformed data, a tensor of shape (n_samples, n_components).

        Returns:
            torch.Tensor: Inverse transformed data, a tensor of shape (n_samples, n_features).

        Raises:
            RuntimeError: If the PCA model has not been fitted yet.
        """
        X = self._to_tensor(X)
        
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        inverse = torch.mm(X, self.components.t()) + self.mean
        return inverse


    def save(self, path):
        """
        Save the PCA model to a file.

        Args:
            path (str): The path to save the model.

        Raises:
            RuntimeError: If the PCA model has not been fitted yet.
        """
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
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
        Converts the input data to a PyTorch tensor.

        Args:
            x: The input data to be converted.

        Returns:
            The converted PyTorch tensor.

        Raises:
            ValueError: If the input data is not a pandas DataFrame, numpy array, or PyTorch tensor.
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

        return x

    @staticmethod
    def load(path):
        """
        Load a saved PCA model from a file.

        Args:
            path (str): The path to the saved model.

        Returns:
            PCA: The loaded PCA model.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the loaded model is not a PCA model.
        """
        checkpoint = torch.load(path)
        model = PCA(checkpoint['n_components'])
        model.mean = checkpoint['mean']
        model.components = checkpoint['components']
        model.singular_values = checkpoint['singular_values']
        model.explained_variance = checkpoint['explained_variance']
        model.explained_variance_ratio = checkpoint['explained_variance_ratio']
        return model
