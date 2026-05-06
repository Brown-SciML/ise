"""GPU-compatible PyTorch scalers for ISEFlow inputs and outputs.

This module provides ``StandardScaler``, ``RobustScaler``, and ``LogScaler``
as ``torch.nn.Module`` subclasses.  They mirror the scikit-learn scaler API
(``fit`` / ``transform`` / ``inverse_transform`` / ``save`` / ``load``) but
operate on ``torch.Tensor`` objects and can be kept on GPU throughout the
forward pass.

Why not use sklearn?
--------------------
Scikit-learn scalers require a CPU round-trip and cannot participate in the
autograd graph.  These subclasses keep scaling arithmetic on whichever device
the model is running on (CUDA or CPU), avoiding expensive device transfers
during inference.

Scalers in the ISEFlow pipeline
--------------------------------
The pretrained ISEFlow models ship a ``scaler_X.pkl`` (sklearn) for input
features and a ``scaler_y.pkl`` (sklearn) for the SLE output target.  These
are **sklearn** scalers used inside ``ise.data.feature_engineer.scale_data``
and ``ISEFlow.predict()``.

The PyTorch scalers in **this** module are used during model training when
GPU-resident tensors must be transformed inside the training loop without
leaving the GPU::

    from ise.data.scaler import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train_tensor)                 # computes mean/std on GPU
    X_scaled = scaler.transform(X_train_tensor)
    X_orig   = scaler.inverse_transform(X_scaled)

    scaler.save("scaler.pt")
    scaler_loaded = StandardScaler.load("scaler.pt")

Scaler summary
--------------
StandardScaler:
    ``(x - mean) / std``.  Zero-variance columns are replaced with a small
    epsilon to prevent division by zero.
RobustScaler:
    ``(x - median) / IQR``.  More resistant to outliers than StandardScaler.
LogScaler:
    ``log(x - min + epsilon)``.  Useful for strictly positive, right-skewed
    targets.  A shift is computed from the training-set minimum so that all
    values remain positive before taking the log.
"""

import torch
from torch import nn

from ise.utils.functions import to_tensor


class StandardScaler(nn.Module):
    """
    A class for scaling input data using mean and standard deviation.

    Args:
        nn.Module: The base class for all neural network modules in PyTorch.

    Attributes:
        mean_ (torch.Tensor): The mean values of the input data.
        scale_ (torch.Tensor): The standard deviation values of the input data.
        device (torch.device): The device (CPU or GPU) on which the calculations are performed.

    Methods:
        fit(X): Computes the mean and standard deviation of the input data.
        transform(X): Scales the input data using the computed mean and standard deviation.
        inverse_transform(X): Reverses the scaling operation on the input data.
        save(path): Saves the mean and standard deviation to a file.
        load(path): Loads the mean and standard deviation from a file.

    """

    def __init__(
        self,
    ):
        super().__init__()
        self.mean_ = None
        self.scale_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def fit(self, X):
        """
        Computes the mean and standard deviation of the input data.

        Args:
            X (torch.Tensor): The input data to be scaled.

        """
        X = to_tensor(X).to(self.device)
        self.mean_ = torch.mean(X, dim=0)
        self.scale_ = torch.std(X, dim=0, unbiased=False)
        self.eps = 1e-8  # to avoid divide by zero
        self.scale_ = torch.where(
            self.scale_ == 0, torch.ones_like(self.scale_) * self.eps, self.scale_
        )  # Avoid division by zero

    def transform(self, X):
        """
        Scales the input data using the computed mean and standard deviation.

        Args:
            X (torch.Tensor): The input data to be scaled.

        Returns:
            torch.Tensor: The scaled input data.

        Raises:
            RuntimeError: If the Scaler instance is not fitted yet.

        """
        X = to_tensor(X).to(self.device)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("This Scaler instance is not fitted yet.")
        transformed = (X - self.mean_) / self.scale_

        # handle NAN (i.e. divide by zero)
        # could also use epsilon value and divide by epsilon instead...
        if torch.isnan(transformed).any():
            transformed = torch.nan_to_num(transformed)

        return transformed

    def inverse_transform(self, X):
        """
        Reverses the scaling operation on the input data.

        Args:
            X (torch.Tensor): The scaled input data to be transformed back.

        Returns:
            torch.Tensor: The transformed input data.

        Raises:
            RuntimeError: If the Scaler instance is not fitted yet.

        """
        X = to_tensor(X).to(self.device)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("This Scaler instance is not fitted yet.")
        return X * self.scale_ + self.mean_

    def save(self, path):
        """
        Saves the mean and standard deviation to a file.

        Args:
            path (str): The path to save the file.

        """
        torch.save(
            {
                "mean_": self.mean_,
                "scale_": self.scale_,
            },
            path,
        )

    @staticmethod
    def load(path):
        """
        Loads the mean and standard deviation from a file.

        Args:
            path (str): The path to load the file from.

        Returns:
            Scaler: A Scaler instance with the loaded mean and standard deviation.

        """
        checkpoint = torch.load(path, weights_only=True)
        scaler = StandardScaler()
        scaler.mean_ = checkpoint["mean_"]
        scaler.scale_ = checkpoint["scale_"]
        return scaler


class RobustScaler(nn.Module):
    """
    A class for scaling input data using the median and interquartile range (IQR),
    making it robust to outliers.

    Args:
        nn.Module: The base class for all neural network modules in PyTorch.

    Attributes:
        median_ (torch.Tensor): The median values of the input data.
        iqr_ (torch.Tensor): The interquartile range (IQR) values of the input data.
        device (torch.device): The device (CPU or GPU) on which the calculations are performed.

    Methods:
        fit(X): Computes the median and IQR of the input data.
        transform(X): Scales the input data using the computed median and IQR.
        inverse_transform(X): Reverses the scaling operation on the input data.
        save(path): Saves the median and IQR to a file.
        load(path): Loads the median and IQR from a file.
    """

    def __init__(self):
        super().__init__()
        self.median_ = None
        self.iqr_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def fit(self, X):
        """
        Computes the median and interquartile range (IQR) of the input data.

        Args:
            X (torch.Tensor): The input data to be scaled.
        """

        X = to_tensor(X).to(self.device)
        self.median_ = torch.median(X, dim=0).values
        q75, q25 = torch.quantile(X, 0.75, dim=0), torch.quantile(X, 0.25, dim=0)
        self.iqr_ = q75 - q25

    def transform(self, X):
        """
        Scales the input data using the computed median and IQR.

        Args:
            X (torch.Tensor): The input data to be scaled.

        Returns:
            torch.Tensor: The scaled input data.

        Raises:
            RuntimeError: If the RobustScaler instance is not fitted yet.
        """

        X = to_tensor(X).to(self.device)
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("This RobustScaler instance is not fitted yet.")
        return (X - self.median_) / (self.iqr_ + 1e-8)

    def inverse_transform(self, X):
        """
        Reverses the scaling operation on the input data.

        Args:
            X (torch.Tensor): The scaled input data to be transformed back.

        Returns:
            torch.Tensor: The transformed input data.

        Raises:
            RuntimeError: If the RobustScaler instance is not fitted yet.
        """

        X = to_tensor(X).to(self.device)
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("This RobustScaler instance is not fitted yet.")
        return X * (self.iqr_ + 1e-8) + self.median_

    def save(self, path):
        torch.save(
            {
                "median_": self.median_,
                "iqr_": self.iqr_,
            },
            path,
        )

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, weights_only=True)
        scaler = RobustScaler()
        scaler.median_ = checkpoint["median_"]
        scaler.iqr_ = checkpoint["iqr_"]
        return scaler


class LogScaler(nn.Module):
    """
    A class for scaling input data using a logarithmic transformation,
    ensuring all values are positive by applying a shift.

    Args:
        epsilon (float, optional): A small constant to avoid log(0) errors. Defaults to 1e-8.

    Attributes:
        epsilon (float): A small constant to avoid log(0) errors.
        min_value (float): The minimum value in the dataset used for shifting.
        device (torch.device): The device (CPU or GPU) on which calculations are performed.

    Methods:
        fit(X): Computes the minimum value of the input data for shifting.
        transform(X): Applies the logarithmic transformation.
        inverse_transform(X): Reverses the log transformation.
        save(path): Saves the scaler parameters to a file.
        load(path): Loads the scaler parameters from a file.
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.min_value = None

    def fit(self, X):
        """
        Computes the minimum value in the dataset to ensure all values remain positive during transformation.

        Args:
            X (torch.Tensor): The input data to be scaled.
        """

        X = to_tensor(X).to(self.device)
        dataset_min = torch.min(X) - self.epsilon
        if dataset_min >= 0:
            self.min_value = 0
        else:
            self.min_value = dataset_min

    def transform(self, X):
        """
        Applies the logarithmic transformation to the input data.

        Args:
            X (torch.Tensor): The input data to be transformed.

        Returns:
            torch.Tensor: The log-transformed input data.
        """

        X = to_tensor(X).to(self.device)
        X_shifted = X - self.min_value  # adding shift (subtracting negative or zero)
        return torch.log(X_shifted + self.epsilon)

    def inverse_transform(self, X):
        """
        Reverses the log transformation to recover the original scale of the data.

        Args:
            X (torch.Tensor): The log-transformed input data.

        Returns:
            torch.Tensor: The transformed input data in its original scale.
        """

        X = to_tensor(X).to(self.device)
        X_exp = torch.exp(X) - self.epsilon
        return X_exp + self.min_value

    def save(self, path):
        torch.save(
            {
                "epsilon": self.epsilon,
                "min_value": self.min_value,
            },
            path,
        )

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, weights_only=True)
        scaler = LogScaler()
        scaler.epsilon = checkpoint["epsilon"]
        scaler.min_value = checkpoint["min_value"]
        return scaler
