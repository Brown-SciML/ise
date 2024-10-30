import torch
from torch import nn
import pandas as pd
import numpy as np

from ise.data.scaler import LogScaler, RobustScaler, StandardScaler

class PCA(nn.Module):
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
        super(PCA, self).__init__()
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.singular_values = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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
        U, S, V = torch.pca_lowrank(
            X_centered,
            q=301,
        )

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

        self.components = V[:, : self.n_components]
        self.singular_values = S[: self.n_components]
        self.explained_variance = explained_variance[: self.n_components]
        self.explained_variance_ratio = explained_variance_ratio[: self.n_components]
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
        torch.save(
            {
                "n_components": self.n_components,
                "mean": self.mean,
                "components": self.components,
                "singular_values": self.singular_values,
                "explained_variance": self.explained_variance,
                "explained_variance_ratio": self.explained_variance_ratio,
            },
            path,
        )

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
        model = PCA(checkpoint["n_components"])
        model.mean = checkpoint["mean"]
        model.components = checkpoint["components"]
        model.singular_values = checkpoint["singular_values"]
        model.explained_variance = checkpoint["explained_variance"]
        model.explained_variance_ratio = checkpoint["explained_variance_ratio"]
        return model


class DimensionProcessor(nn.Module):
    """
    A class that performs dimension processing using PCA and scaling.

    Args:
        pca_model (str or PCA): The PCA model to use for dimension reduction. It can be either a path to a saved PCA model or an instance of the PCA class.
        scaler_model (str or Scaler): The scaler model to use for scaling the data. It can be either a path to a saved scaler model or an instance of the scaler class.
        scaler_method (str): The method to use for scaling. Must be one of 'standard', 'robust', or 'log'.

    Attributes:
        device (str): The device to use for computation. It is set to 'cuda' if a CUDA-enabled GPU is available, otherwise it is set to 'cpu'.
        pca (PCA): The PCA model used for dimension reduction.
        scaler (Scaler): The scaler model used for scaling the data.

    Raises:
        ValueError: If the `pca_model` is not a valid path or instance of PCA, or if the `scaler_model` is not a valid path or instance of the scaler class.
        RuntimeError: If the PCA model has not been fitted yet.

    """

    def __init__(
        self,
        pca_model,
        scaler_model,
        scaler_method="standard",
    ):
        super(DimensionProcessor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # LOAD PCA
        if isinstance(pca_model, str):
            self.pca = PCA.load(pca_model)
        elif isinstance(pca_model, PCA):
            self.pca = pca_model
        else:
            raise ValueError("pca_model must be a path (str) or a PCA instance")
        if self.pca.mean is None or self.pca.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")

        # LOAD SCALER
        if scaler_method == "standard":
            scaler_class = StandardScaler
        elif scaler_method == "robust":
            scaler_class = RobustScaler
        elif scaler_method == "log":
            scaler_class = LogScaler
        else:
            raise ValueError("scaler_method must be 'standard', 'robust', or 'log'")

        if isinstance(scaler_model, str):
            self.scaler = scaler_class.load(scaler_model)
        elif isinstance(scaler_model, scaler_class):
            self.scaler = scaler_model
        else:
            raise ValueError("pca_model must be a path (str) or a PCA instance")
        # if self.scaler.mean_ is None or self.scaler.scale_ is None:
        #     raise RuntimeError("This StandardScalerPyTorch instance is not fitted yet.")

        self.scaler.to(self.device)
        self.pca.to(self.device)
        self.to(self.device)

    def to_pca(self, data):
        """
        Transforms the input data to the PCA space.

        Args:
            data (torch.Tensor or pd.DataFrame): The input data to transform.

        Returns:
            torch.Tensor: The transformed data in the PCA space.

        """
        data = data.to(self.device)
        scaled = self.scaler.transform(data)  # scale
        return self.pca.transform(scaled)  # convert to pca

    def to_grid(self, pcs, unscale=True):
        """
        Transforms the input principal components (pcs) to the original data space.

        Args:
            pcs (torch.Tensor or pd.DataFrame): The principal components to transform.
            unscale (bool): Whether to unscale the transformed data. If True, the data will be unscaled using the scaler model.

        Returns:
            torch.Tensor or pd.DataFrame: The transformed data in the original data space.

        """
        if not isinstance(pcs, torch.Tensor):
            if isinstance(pcs, pd.DataFrame):
                pcs = pcs.values
            pcs = torch.tensor(pcs, dtype=torch.float32).to(self.device)
        else:
            pcs = pcs.to(self.device)
        # Ensure components and mean are on the same device as pcs
        components = self.pca.components.to(self.device)
        pca_mean = self.pca.mean.to(self.device)
        # Now, the operation should not cause a device mismatch error
        scaled_grid = torch.mm(pcs, components.t()) + pca_mean

        if unscale:
            return self.scaler.inverse_transform(scaled_grid)
            # scale = self.scaler.scale_.to(self.device)
            # scaler_mean = self.scaler.mean_.to(self.device)
            # unscaled_grid = scaled_grid * scale + scaler_mean
            # return unscaled_grid

        return scaled_grid
