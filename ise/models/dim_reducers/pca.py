import torch
from torch import nn
import pandas as pd
import numpy as np

from ise.data.scaler import LogScaler, RobustScaler, StandardScaler

class PCA(nn.Module):
    """
    Principal Component Analysis (PCA) using PyTorch.

    This class performs Principal Component Analysis (PCA) for dimensionality 
    reduction, leveraging PyTorch's built-in operations. The model can be 
    trained on input data to extract principal components and later transform 
    new data into the reduced-dimensional space.

    Attributes:
        n_components (int or float): Number of principal components to keep. 
            If an integer, it represents the exact number of components. 
            If a float between 0 and 1, it represents the proportion of variance to retain.
        mean (torch.Tensor): Mean of the input data, computed during fitting.
        components (torch.Tensor): Principal components (eigenvectors) of the fitted data.
        singular_values (torch.Tensor): Singular values from singular value decomposition.
        explained_variance (torch.Tensor): Variance explained by each principal component.
        explained_variance_ratio (torch.Tensor): Proportion of total variance explained by each component.
        device (str): The computation device ('cuda' if available, otherwise 'cpu').
    """

    def __init__(self, n_components):
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
        Fits the PCA model to the input data.

        Computes the principal components, singular values, explained variance, 
        and variance ratios using singular value decomposition.

        Args:
            X (np.ndarray or pd.DataFrame): Input data of shape (n_samples, n_features).

        Returns:
            PCA: The fitted PCA model instance.

        Raises:
            ValueError: If `n_components` is neither a valid float (0,1) nor an integer.
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
        Transforms input data into the principal component space.

        Args:
            X (np.ndarray or pd.DataFrame): Input data of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Transformed data in the principal component space, 
            with shape (n_samples, n_components).

        Raises:
            RuntimeError: If the PCA model has not been fitted.
        """

        X = self._to_tensor(X)
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        X_centered = X - self.mean
        return torch.mm(X_centered, self.components)

    def inverse_transform(self, X):
        """
        Reconstructs the original data from the principal component space.

        Args:
            X (np.ndarray or pd.DataFrame): Transformed data in the reduced space,
                with shape (n_samples, n_components).

        Returns:
            torch.Tensor: Reconstructed data in the original feature space.

        Raises:
            RuntimeError: If the PCA model has not been fitted.
        """

        X = self._to_tensor(X)

        if self.mean is None or self.components is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        inverse = torch.mm(X, self.components.t()) + self.mean
        return inverse

    def save(self, path):
        """
        Saves the fitted PCA model to a specified file.

        Args:
            path (str): File path where the model should be saved.

        Raises:
            RuntimeError: If the PCA model has not been fitted before saving.
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
        Converts input data into a PyTorch tensor.

        Args:
            x (np.ndarray, pd.DataFrame, or torch.Tensor): Input data.

        Returns:
            torch.Tensor: Converted data as a PyTorch tensor.

        Raises:
            ValueError: If input data type is not supported.
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
        Loads a previously saved PCA model.

        Args:
            path (str): Path to the saved PCA model.

        Returns:
            PCA: Loaded PCA model instance.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            RuntimeError: If the loaded model is not a valid PCA instance.
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
    A pipeline for dimensionality reduction using PCA and data scaling.

    This class integrates a PCA model for dimensionality reduction and a 
    scaling method for normalization. It facilitates transformation between 
    the original data space and the reduced PCA space.

    Args:
        pca_model (str or PCA): A pre-trained PCA model or a file path to a saved PCA model.
        scaler_model (str or StandardScaler, RobustScaler, LogScaler): 
            A scaler model instance or a file path to a saved scaler model.
        scaler_method (str, optional): Scaling method to use. 
            Must be one of 'standard', 'robust', or 'log'. Defaults to 'standard'.

    Attributes:
        device (str): Computation device ('cuda' if available, otherwise 'cpu').
        pca (PCA): PCA model used for dimensionality reduction.
        scaler (StandardScaler, RobustScaler, or LogScaler): Scaling model for data normalization.

    Raises:
        ValueError: If the `pca_model` or `scaler_model` are invalid.
        RuntimeError: If the PCA model has not been fitted.
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
        Applies PCA transformation to the input data.

        Args:
            data (torch.Tensor or pd.DataFrame): Input data for transformation.

        Returns:
            torch.Tensor: Data transformed into the PCA space.
        """

        data = data.to(self.device)
        scaled = self.scaler.transform(data)  # scale
        return self.pca.transform(scaled)  # convert to pca

    def to_grid(self, pcs, unscale=True):
        """
        Converts PCA-transformed data back to the original feature space.

        Args:
            pcs (torch.Tensor or pd.DataFrame): Principal components to transform.
            unscale (bool, optional): Whether to apply inverse scaling. Defaults to True.

        Returns:
            torch.Tensor or pd.DataFrame: Reconstructed data in the original feature space.
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
