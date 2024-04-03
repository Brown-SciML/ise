"""Description

Classes:
    - PCA: Class for Prinicpal Component Analysis, including fitting and transforming data.
    - DimensionProcessor: Class for dimension processing using PCA and scaling.
    - WeakPredictor: Class for an individual 'weak' predictor model in a deep ensemble.
    - DeepEnsemble: Class for a deep ensemble of WeakPredictor models.
    - NormalizingFlow: Class for a Normalizing Flow model.
    - HybridEmulator: Model class for emulating ismip6 ice sheet models while quantifying both data and model uncertainty.
    
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
from nflows import distributions, flows, transforms
from torch import nn, optim

from ise.data.dataclasses import EmulatorDataset
from ise.data.scaler import LogScaler, RobustScaler, StandardScaler
from ise.models.loss import MSEDeviationLoss, WeightedMSELoss
from ise.utils.functions import to_tensor


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


class WeakPredictor(nn.Module):
    """
    A class representing a weak predictor model.

    Args:
        lstm_num_layers (int): The number of LSTM layers.
        lstm_hidden_size (int): The hidden size of the LSTM layers.
        input_size (int, optional): The input size of the model. Defaults to 43.
        output_size (int, optional): The output size of the model. Defaults to 1.
        dim_processor (DimensionProcessor or str, optional): The dimension processor object or path to a PCA object. Defaults to None.
        scaler_path (str, optional): The path to a scaler object. Required if dim_processor is a path to a PCA object. Defaults to None.
        ice_sheet (str, optional): The ice sheet type. Defaults to "AIS".
        criterion (torch.nn.Module, optional): The loss criterion. Defaults to torch.nn.MSELoss().

    Attributes:
        lstm_num_layers (int): The number of LSTM layers.
        lstm_num_hidden (int): The hidden size of the LSTM layers.
        input_size (int): The input size of the model.
        output_size (int): The output size of the model.
        ice_sheet (str): The ice sheet type.
        ice_sheet_dim (tuple): The dimensions of the ice sheet.
        device (str): The device used for computation.
        lstm (torch.nn.LSTM): The LSTM layer.
        relu (torch.nn.ReLU): The ReLU activation function.
        linear1 (torch.nn.Linear): The first linear layer.
        linear_out (torch.nn.Linear): The output linear layer.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        dropout (torch.nn.Dropout): The dropout layer.
        criterion (torch.nn.Module): The loss criterion.
        trained (bool): Indicates if the model has been trained.
        dim_processor (DimensionProcessor or None): The dimension processor object.

    Methods:
        forward(x): Performs a forward pass through the model.
        fit(X, y, epochs, sequence_length, batch_size, loss, val_X, val_y): Trains the model.
        predict(X, sequence_length, batch_size): Makes predictions using the trained model.
    """

    def __init__(
        self,
        lstm_num_layers,
        lstm_hidden_size,
        input_size=83,
        output_size=1,
        dim_processor=None,
        scaler_path=None,
        ice_sheet="AIS",
        criterion=torch.nn.MSELoss(),
        projection_length=86,
    ):
        super(WeakPredictor, self).__init__()

        # Initialize attributes
        self.lstm_num_layers = int(lstm_num_layers)
        self.lstm_num_hidden = int(lstm_hidden_size)
        self.input_size = input_size
        self.output_size = output_size
        self.ice_sheet = ice_sheet
        self.projection_length = projection_length
        self.ice_sheet_dim = (761, 761) if ice_sheet == "AIS" else (337, 577)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        # Initialize model layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=int(lstm_hidden_size),
            batch_first=True,
            num_layers=lstm_num_layers,
        )
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=lstm_hidden_size, out_features=32)
        self.linear_out = nn.Linear(in_features=32, out_features=output_size)

        # Initialize optimizer and other components
        self.optimizer = optim.Adam(self.parameters())
        self.dropout = nn.Dropout(p=0.2)
        self.criterion = criterion
        self.trained = False

        # Initialize dimension processor
        if isinstance(dim_processor, DimensionProcessor):
            self.dim_processor = dim_processor.to(self.device)
        elif isinstance(dim_processor, str) and scaler_path is None:
            raise ValueError(
                "If dim_processor is a path to a PCA object, scaler_path must be provided"
            )
        elif isinstance(dim_processor, str) and scaler_path is not None:
            self.dim_processor = DimensionProcessor(
                pca_model=self.pca_model, scaler_model=scaler_path
            ).to(self.device)
        elif dim_processor is None:
            self.dim_processor = None
        else:
            raise ValueError(
                "dim_processor must be a DimensionProcessor instance or a path (str) to a PCA object with scaler_path specified as a Scaler object."
            )

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Perform LSTM forward pass
        batch_size = x.shape[0]
        h0 = (
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_num_hidden)
            .requires_grad_()
            .to(self.device)
        )
        c0 = (
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_num_hidden)
            .requires_grad_()
            .to(self.device)
        )
        _, (hn, _) = self.lstm(x, (h0, c0))
        x = hn[-1, :, :]

        # Perform linear layer operations
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear_out(x)

        return x

    def fit(
        self, X, y, epochs=100, sequence_length=5, batch_size=64, loss=None, val_X=None, val_y=None
    ):
        """
        Trains the model.

        Args:
            X (numpy.ndarray or pandas.DataFrame): The input data.
            y (numpy.ndarray or pandas.DataFrame): The target data.
            epochs (int, optional): The number of epochs to train for. Defaults to 100.
            sequence_length (int, optional): The sequence length for creating input sequences. Defaults to 5.
            batch_size (int, optional): The batch size. Defaults to 64.
            loss (torch.nn.Module, optional): The loss function to use. If None, the default criterion is used. Defaults to None.
            val_X (numpy.ndarray or pandas.DataFrame, optional): The validation input data. Defaults to None.
            val_y (numpy.ndarray or pandas.DataFrame, optional): The validation target data. Defaults to None.
        """
        # Convert data to tensors and move to device
        X, y = to_tensor(X).to(self.device), to_tensor(y).to(self.device)

        # Check if validation data is provided
        if val_X is not None and val_y is not None:
            validate = True
        else:
            validate = False

        # Set loss criterion
        if loss is not None:
            self.criterion = loss.to(self.device)
        elif loss is None and self.criterion is None:
            raise ValueError("loss must be provided if criterion is None.")
        self.criterion = self.criterion.to(self.device)

        # Convert data to numpy arrays if pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        # Create dataset and data loader
        dataset = EmulatorDataset(X, y, sequence_length=sequence_length, projection_length=self.projection_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set model to training mode
        self.train()
        self.to(self.device)

        # Training loop
        for epoch in range(1, epochs + 1):
            self.train()
            batch_losses = []
            for i, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())

            # Print average batch loss and validation loss (if provided)
            if validate:
                val_preds = self.predict(
                    val_X, sequence_length=sequence_length, batch_size=batch_size
                ).to(self.device)
                val_loss = self.criterion(val_preds, torch.tensor(val_y, device=self.device))
                print(
                    f"Epoch {epoch}, Average Batch Loss: {sum(batch_losses) / len(batch_losses)}, Validation Loss: {val_loss}"
                )
            else:
                average_batch_loss = sum(batch_losses) / len(batch_losses)
                print(f"Epoch {epoch}, Average Batch Loss: {average_batch_loss}")

        self.trained = True

    def predict(self, X, sequence_length=5, batch_size=64):
        """
        Makes predictions using the trained model.

        Args:
            X (numpy.ndarray or pandas.DataFrame): The input data.
            sequence_length (int, optional): The sequence length for creating input sequences. Defaults to 5.
            batch_size (int, optional): The batch size. Defaults to 64.

        Returns:
            torch.Tensor: The predicted values.
        """
        # Set model to evaluation mode
        self.eval()
        self.to(self.device)

        # Convert data to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Create dataset and data loader
        dataset = EmulatorDataset(X, y=None, sequence_length=sequence_length, projection_length=self.projection_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Move input data to device
        X = to_tensor(X).to(self.device)

        preds = torch.tensor([]).to(self.device)
        for X_test_batch in data_loader:
            self.eval()
            X_test_batch = X_test_batch.to(self.device)
            y_pred = self.forward(X_test_batch)
            preds = torch.cat((preds, y_pred), 0)

        return preds


class DeepEnsemble(nn.Module):
    """
    A deep ensemble model for prediction tasks.

    Args:
    - weak_predictors (list, optional): List of WeakPredictor instances. If not provided, weak predictors will be randomly generated.
    - forcing_size (int, optional): Size of the input forcing data. Required if weak_predictors is not provided.
    - sle_size (int, optional): Size of the input SLE data. Required if weak_predictors is not provided.
    - num_predictors (int, optional): Number of weak predictors to generate if weak_predictors is not provided.

    Attributes:
    - weak_predictors (list): List of WeakPredictor instances.
    - trained (bool): Indicates whether all weak predictors have been trained.

    Methods:
    - forward(x): Performs forward pass through the model.
    - predict(x): Makes predictions using the model.
    - fit(X, y, epochs=100, batch_size=64, sequence_length=5): Trains the model.
    - save(model_path): Saves the model parameters and metadata.
    - load(model_path): Loads the model parameters and metadata and returns an instance of the model.
    """

    def __init__(self, weak_predictors: list = [], forcing_size=83, sle_size=1, num_predictors=3, projection_length=86):
        super(DeepEnsemble, self).__init__()
        self.forcing_size = forcing_size + 1 # for latent z from nf
        self.sle_size = sle_size
        self.projection_length = projection_length
        
        if not weak_predictors:
            if forcing_size is None or sle_size is None:
                raise ValueError(
                    "forcing_size and sle_size must be provided if weak_predictors is not provided"
                )
            self.loss_choices = [
                torch.nn.MSELoss(),
                torch.nn.MSELoss(),
                torch.nn.L1Loss(),
                torch.nn.HuberLoss(),
            ]
            # loss_probabilities = [.45, .05, .3, .2]
            self.weak_predictors = [
                WeakPredictor(
                    lstm_num_layers=np.random.randint(low=1, high=3, size=1)[0],
                    lstm_hidden_size=np.random.choice([512, 256, 128, 64], 1)[0],
                    criterion=np.random.choice(
                        self.loss_choices,
                        1,
                    )[0],
                    input_size=self.forcing_size,
                    output_size=self.sle_size,
                    projection_length=self.projection_length,
                )
                for _ in range(num_predictors)
            ]
        else:
            if isinstance(weak_predictors, list):
                self.weak_predictors = weak_predictors
                if not all([isinstance(x, WeakPredictor) for x in weak_predictors]):
                    raise ValueError("weak_predictors must be a list of WeakPredictor instances")
            else:
                raise ValueError("weak_predictors must be a list of WeakPredictor instances")

            if any([x for x in weak_predictors if not isinstance(x, WeakPredictor)]):
                raise ValueError("weak_predictors must be a list of WeakPredictor instances")

        # check to see if all weak predictors are trained
        self.trained = all([wp.trained for wp in self.weak_predictors])

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
        - x: Input data.

        Returns:
        - mean_prediction: Mean prediction of the ensemble.
        - epistemic_uncertainty: Epistemic uncertainty of the ensemble.
        """
        if not self.trained:
            warnings.warn("This model has not been trained. Predictions will not be accurate.")
        preds = torch.stack([wp.predict(x) for wp in self.weak_predictors], axis=1)
        mean_prediction = torch.mean(preds, axis=1).squeeze()
        epistemic_uncertainty = torch.std(preds, axis=1).squeeze()
        return mean_prediction, epistemic_uncertainty

    def predict(self, x):
        """
        Makes predictions using the model.

        Args:
        - x: Input data.

        Returns:
        - predictions: Predictions made by the model.
        """
        self.eval()
        return self.forward(x)

    def fit(
        self,
        X,
        y,
        epochs=100,
        batch_size=64,
        sequence_length=5,
    ):
        """
        Trains the model.

        Args:
        - X: Input data.
        - y: Target data.
        - epochs (int, optional): Number of epochs to train the model. Default is 100.
        - batch_size (int, optional): Batch size for training. Default is 64.
        - sequence_length (int, optional): Length of input sequences. Default is 5.
        """
        if self.trained:
            warnings.warn("This model has already been trained. Training anyways.")
        for i, wp in enumerate(self.weak_predictors):
            print(f"Training Weak Predictor {i+1} of {len(self.weak_predictors)}:")
            wp.fit(X, y, epochs=epochs, batch_size=batch_size, sequence_length=sequence_length)
            print("")
        self.trained = True

    def save(self, model_path):
        """
        Saves the model parameters and metadata, including each WeakPredictor individually.

        Args:
        - model_path: Path to save the model.
        """
        if not self.trained:
            raise ValueError(
                "This model has not been trained yet. Please train the model before saving."
            )

        # Create a directory for the weak_predictor models if it does not exist
        model_dir = os.path.dirname(model_path)
        weak_predictors_dir = os.path.join(model_dir, "weak_predictors")
        if not os.path.exists(weak_predictors_dir):
            os.makedirs(weak_predictors_dir)

        # Metadata for the ensemble, not including the state_dicts of the weak_predictors
        metadata = {
            "model_type": self.__class__.__name__,
            "weak_predictors": [
                {
                    "lstm_num_layers": int(wp.lstm_num_layers),
                    "lstm_num_hidden": int(wp.lstm_num_hidden),
                    "criterion": wp.criterion.__class__.__name__,
                    "forcing_size": wp.input_size,
                    "sle_size": wp.output_size,
                    "trained": wp.trained,
                    "weak_predictor_path": os.path.join(
                        "weak_predictors", f"weak_predictor_{i+1}.pth"
                    ),  # Path relative to model_dir
                }
                for i, wp in enumerate(self.weak_predictors)
            ],
        }
        metadata_path = model_path.replace(".pth", "_metadata.json")
        with open(metadata_path, "w") as file:
            json.dump(metadata, file, indent=4)
        print(f"Model metadata saved to {metadata_path}")

        # Save the state dictionary of the DeepEnsemble (excluding weak_predictors' state_dicts)
        torch.save(self.state_dict(), model_path)
        print(f"Model parameters saved to {model_path}")

        # Save each WeakPredictor model separately
        for i, wp in enumerate(self.weak_predictors):
            wp_model_path = os.path.join(weak_predictors_dir, f"weak_predictor_{i+1}.pth")
            torch.save(wp.state_dict(), wp_model_path)
            print(f"WeakPredictor {i+1} model parameters saved to {wp_model_path}")

    @classmethod
    def load(cls, model_path):
        """
        Loads the model architecture metadata from a JSON file and the model parameters,
        reconstructs the model, and returns an instance of the model.

        Parameters:
        - model_path: Path to the file from which model parameters should be loaded.

        Returns:
        - An instance of the model with loaded parameters.
        """
        # Load metadata
        metadata_path = model_path.replace(".pth", "_metadata.json")
        with open(metadata_path, "r") as file:
            metadata = json.load(file)

        # Check for correct model type
        if cls.__name__ != metadata["model_type"]:
            raise ValueError(
                f"Model type in metadata ({metadata['model_type']}) does not match the class type ({cls.__name__})"
            )

        # Prepare loss lookup for instantiating weak predictors
        loss_lookup = {
            "MSELoss": torch.nn.MSELoss(),
            "L1Loss": torch.nn.L1Loss(),
            "HuberLoss": torch.nn.HuberLoss(),
        }

        # Load weak_predictors
        model_dir = os.path.dirname(model_path)
        weak_predictors = []
        for wp_metadata in metadata["weak_predictors"]:
            wp_path = os.path.join(model_dir, wp_metadata["weak_predictor_path"])
            criterion = loss_lookup[wp_metadata["criterion"]]
            wp = WeakPredictor(
                lstm_num_layers=wp_metadata["lstm_num_layers"],
                lstm_hidden_size=wp_metadata["lstm_num_hidden"],
                input_size=wp_metadata["forcing_size"],
                output_size=wp_metadata["sle_size"],
                criterion=criterion,
            )
            if torch.cuda.is_available():
                wp.load_state_dict(torch.load(wp_path))
            else:
                wp.load_state_dict(torch.load(wp_path, map_location="cpu"))
            wp.eval()  # Set the weak predictor to evaluation mode
            weak_predictors.append(wp)

        # Instantiate the model with the loaded weak predictors
        model = cls(weak_predictors=weak_predictors)

        # Load the ensemble model parameters excluding weak predictors as they are loaded separately
        ensemble_state_dict = torch.load(model_path)
        # It might be necessary to filter out weak_predictor states from the ensemble state_dict if they were included
        model.load_state_dict(ensemble_state_dict, strict=False)

        model.forcing_size = wp_metadata["forcing_size"]
        model.sle_size = wp_metadata["sle_size"]
        model.num_predictors = len(weak_predictors)
        model.eval()  # Set the model to evaluation mode

        return model


class NormalizingFlow(nn.Module):
    """
    A class representing a Normalizing Flow model.

    Args:
        forcing_size (int): The size of the forcing input features.
        sle_size (int): The size of the predicted SLE (Stochastic Lagrangian Ensemble) output.

    Attributes:
        num_flow_transforms (int): The number of flow transforms in the model.
        num_input_features (int): The number of input features.
        num_predicted_sle (int): The number of predicted SLE features.
        flow_hidden_features (int): The number of hidden features in the flow.
        device (str): The device used for computation (either "cuda" or "cpu").
        base_distribution (distributions.normal.ConditionalDiagonalNormal): The base distribution for the flow.
        t (transforms.base.CompositeTransform): The composite transform for the flow.
        flow (flows.base.Flow): The flow model.
        optimizer (optim.Adam): The optimizer used for training the flow.
        criterion (callable): The criterion used for calculating the log probability of the flow.
        trained (bool): Indicates whether the model has been trained or not.

    Methods:
        fit(X, y, epochs=100, batch_size=64): Trains the model on the given input and output data.
        sample(features, num_samples, return_type="numpy"): Generates samples from the model.
        get_latent(x, latent_constant=0.0): Computes the latent representation of the input data.
        aleatoric(features, num_samples): Computes the aleatoric uncertainty of the model predictions.
        save(path): Saves the trained model to the specified path.
    """

    def __init__(
        self,
        forcing_size=43,
        sle_size=1,
        projection_length=86,
    ):
        super(NormalizingFlow, self).__init__()
        self.num_flow_transforms = 5
        self.num_input_features = forcing_size
        self.num_predicted_sle = sle_size
        self.flow_hidden_features = sle_size * 2
        self.projection_length=projection_length,
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        self.base_distribution = distributions.normal.ConditionalDiagonalNormal(
            shape=[self.num_predicted_sle],
            context_encoder=nn.Linear(self.num_input_features, self.flow_hidden_features),
        )

        t = []
        for _ in range(self.num_flow_transforms):
            t.append(
                transforms.permutations.RandomPermutation(
                    features=self.num_predicted_sle,
                )
            )
            t.append(
                transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                    features=self.num_predicted_sle,
                    hidden_features=self.flow_hidden_features,
                    context_features=self.num_input_features,
                )
            )

        self.t = transforms.base.CompositeTransform(t)

        self.flow = flows.base.Flow(transform=self.t, distribution=self.base_distribution)

        self.optimizer = optim.Adam(self.flow.parameters())
        self.criterion = self.flow.log_prob
        self.trained = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def fit(self, X, y, epochs=100, batch_size=64):
        """
        Trains the model on the given input and output data.

        Args:
            X (array-like): The input data.
            y (array-like): The output data.
            epochs (int): The number of training epochs (default: 100).
            batch_size (int): The batch size for training (default: 64).
        """
        X, y = to_tensor(X).to(self.device), to_tensor(y).to(self.device)
        dataset = EmulatorDataset(X, y, sequence_length=1, projection_length=self.projection_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = []
            for i, (x, y) in enumerate(data_loader):
                x = x.to(self.device).view(x.shape[0], -1)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                loss = torch.mean(-self.flow.log_prob(inputs=y, context=x))
                if torch.isnan(loss):
                    stop = ""
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
            print(f"Epoch {epoch}, Loss: {sum(epoch_loss) / len(epoch_loss)}")
        self.trained = True

    def sample(self, features, num_samples, return_type="numpy"):
        """
        Generates samples from the model.

        Args:
            features (array-like): The input features for generating samples.
            num_samples (int): The number of samples to generate.
            return_type (str): The return type of the samples ("numpy" or "tensor", default: "numpy").

        Returns:
            array-like or torch.Tensor: The generated samples.
        """
        if not isinstance(features, torch.Tensor):
            features = to_tensor(features)
        samples = self.flow.sample(num_samples, context=features).reshape(
            features.shape[0], num_samples
        )
        if return_type == "tensor":
            pass
        elif return_type == "numpy":
            samples = samples.detach().cpu().numpy()
        else:
            raise ValueError("return_type must be 'numpy' or 'tensor'")
        return samples

    def get_latent(self, x, latent_constant=0.0):
        """
        Computes the latent representation of the input data.

        Args:
            x (array-like): The input data.
            latent_constant (float): The constant value for the latent representation (default: 0.0).

        Returns:
            torch.Tensor: The latent representation of the input data.
        """
        x = to_tensor(x).to(self.device)
        latent_constant_tensor = torch.ones((x.shape[0], 1)).to(self.device) * latent_constant
        z, _ = self.t(latent_constant_tensor.float(), context=x)
        return z

    def aleatoric(self, features, num_samples):
        """
        Computes the aleatoric uncertainty of the model predictions.

        Args:
            features (array-like): The input features for computing the uncertainty.
            num_samples (int): The number of samples to use for computing the uncertainty.

        Returns:
            array-like: The aleatoric uncertainty of the model predictions.
        """
        if not isinstance(features, torch.Tensor):
            features = to_tensor(features)
        samples = self.flow.sample(num_samples, context=features)
        samples = samples.detach().cpu().numpy()
        std = np.std(samples, axis=1).squeeze()
        return std

    def save(self, path):
        """
        Saves the model parameters and metadata to the specified path.

        Args:
            path (str): The path to save the model.
        """

        if not self.trained:
            raise ValueError(
                "This model has not been trained yet. Please train the model before saving."
            )
        # Prepare metadata for saving
        metadata = {
            "forcing_size": self.num_input_features,
            "sle_size": self.num_predicted_sle,
        }
        metadata_path = path + "_metadata.json"

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Save model parameters
        torch.save(self.state_dict(), path)
        print(f"Model and metadata saved to {path} and {metadata_path}, respectively.")

    @staticmethod
    def load(path):
        """
        Loads the NormalizingFlow model from the specified path.

        Args:
            path (str): The path to load the model from.

        Returns:
            NormalizingFlow: The loaded NormalizingFlow model.
        """
        # Load metadata
        metadata_path = path + "_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Reconstruct the model using the loaded metadata
        model = NormalizingFlow(
            forcing_size=metadata["forcing_size"], sle_size=metadata["sle_size"]
        )

        # Load the model parameters
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(path))
        model.eval()  # Set the model to evaluation model

        return model


class HybridEmulator(torch.nn.Module):
    """
    A hybrid emulator that combines a deep ensemble and a normalizing flow model.

    Args:
        deep_ensemble (DeepEnsemble): The deep ensemble model.
        normalizing_flow (NormalizingFlow): The normalizing flow model.

    Attributes:
        device (str): The device used for computation (cuda or cpu).
        deep_ensemble (DeepEnsemble): The deep ensemble model.
        normalizing_flow (NormalizingFlow): The normalizing flow model.
        trained (bool): Indicates whether the model has been trained.

    Methods:
        fit(X, y, epochs=100, nf_epochs=None, de_epochs=None, sequence_length=5):
            Fits the hybrid emulator to the training data.
        forward(x):
            Performs a forward pass through the hybrid emulator.
        save(save_dir):
            Saves the trained model to the specified directory.
        load(deep_ensemble_path, normalizing_flow_path):
            Loads a trained model from the specified paths.

    """

    def __init__(self, deep_ensemble, normalizing_flow,):
        super(HybridEmulator, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        if not isinstance(deep_ensemble, DeepEnsemble):
            raise ValueError("deep_ensemble must be a DeepEmulator instance")
        if not isinstance(normalizing_flow, NormalizingFlow):
            raise ValueError("normalizing_flow must be a NormalizingFlow instance")

        self.deep_ensemble = deep_ensemble.to(self.device)
        self.normalizing_flow = normalizing_flow.to(self.device)
        self.trained = self.deep_ensemble.trained and self.normalizing_flow.trained
        self.scaler_path = None

    def fit(self, X, y, epochs=100, nf_epochs=None, de_epochs=None, sequence_length=5):
        """
        Fits the hybrid emulator to the training data.

        Args:
            X (array-like): The input training data.
            y (array-like): The target training data.
            epochs (int): The number of epochs to train the model (default: 100).
            nf_epochs (int): The number of epochs to train the normalizing flow model (default: None).
                If not specified, the same number of epochs as the overall model will be used.
            de_epochs (int): The number of epochs to train the deep ensemble model (default: None).
                If not specified, the same number of epochs as the overall model will be used.
            sequence_length (int): The sequence length used for training the deep ensemble model (default: 5).

        """
        torch.manual_seed(np.random.randint(0, 100000))
        # if specific epoch numbers are not supplied, use the same number of epochs for both
        if nf_epochs is None:
            nf_epochs = epochs
        if de_epochs is None:
            de_epochs = epochs

        X, y = to_tensor(X).to(self.device), to_tensor(y).to(self.device)
        if self.trained:
            warnings.warn("This model has already been trained. Training anyways.")
        if not self.normalizing_flow.trained:
            print(f"\nTraining Normalizing Flow ({nf_epochs} epochs):")
            self.normalizing_flow.fit(X, y, epochs=nf_epochs)
        z = self.normalizing_flow.get_latent(
            X,
        ).detach()
        X_latent = torch.concatenate((X, z), axis=1)
        if not self.deep_ensemble.trained:
            print(f"\nTraining Deep Ensemble ({de_epochs} epochs):")
            self.deep_ensemble.fit(X_latent, y, epochs=de_epochs, sequence_length=sequence_length)
        self.trained = True

    def forward(
        self,
        x,
        smooth_projection=False,
    ):
        """
        Performs a forward pass through the hybrid emulator.

        Args:
            x (array-like): The input data.

        Returns:
            tuple: A tuple containing the prediction, epistemic uncertainty, and aleatoric uncertainty.

        """
        self.eval()
        x = to_tensor(x).to(self.device)
        if not self.trained:
            warnings.warn("This model has not been trained. Predictions will not be accurate.")
        z = self.normalizing_flow.get_latent(
            x,
        ).detach()
        X_latent = torch.concatenate((x, z), axis=1)
        prediction, epistemic = self.deep_ensemble(X_latent)
        aleatoric = self.normalizing_flow.aleatoric(x, 100)
        prediction = prediction.detach().cpu().numpy()
        epistemic = epistemic.detach().cpu().numpy()
        uncertainties = dict(
            total=aleatoric + epistemic,
            epistemic=epistemic,
            aleatoric=aleatoric,
        )

        if smooth_projection:
            stop = ""
        return prediction, uncertainties

    def predict(self, x, output_scaler=None, smooth_projection=False):
        self.eval()
        if output_scaler is None and self.scaler_path is None:
            warnings.warn("No scaler path provided, uncertainties are not in units of SLE.")
            return self.forward(x, smooth_projection=smooth_projection)
        if not isinstance(output_scaler, str):
            if 'fit' not in dir(output_scaler) or 'transform' not in dir(output_scaler):
                raise ValueError("output_scaler must be a Scaler object or a path to a Scaler object.")
        else:
            self.scaler_path = output_scaler
            with open(self.scaler_path, "rb") as f:
                output_scaler = pickle.load(f)

        predictions, uncertainties = self.forward(x, smooth_projection=smooth_projection)
        epi = uncertainties["epistemic"]
        ale = uncertainties["aleatoric"]

        bound_epistemic, bound_aleatoric = predictions + epi, predictions + ale

        unscaled_predictions = output_scaler.inverse_transform(predictions.reshape(-1, 1))
        unscaled_bound_epistemic = output_scaler.inverse_transform(bound_epistemic.reshape(-1, 1))
        unscaled_bound_aleatoric = output_scaler.inverse_transform(bound_aleatoric.reshape(-1, 1))
        epistemic = unscaled_bound_epistemic - unscaled_predictions
        aleatoric = unscaled_bound_aleatoric - unscaled_predictions

        uncertainties = dict(
            total=epistemic + aleatoric,
            epistemic=epistemic,
            aleatoric=aleatoric,
        )

        return unscaled_predictions, uncertainties

    def save(self, save_dir):
        """
        Saves the trained model to the specified directory.

        Args:
            save_dir (str): The directory to save the model.

        Raises:
            ValueError: If the model has not been trained yet or if save_dir is a file.

        """
        if not self.trained:
            raise ValueError(
                "This model has not been trained yet. Please train the model before saving."
            )
        if save_dir.endswith(".pth"):
            raise ValueError("save_dir must be a directory, not a file")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.deep_ensemble.save(f"{save_dir}/deep_ensemble.pth")
        self.normalizing_flow.save(f"{save_dir}/normalizing_flow.pth")

    @staticmethod
    def load(model_dir=None, deep_ensemble_path=None, normalizing_flow_path=None):
        """
        Loads a trained model from the specified paths.

        Args:
            deep_ensemble_path (str): The path to the saved deep ensemble model.
            normalizing_flow_path (str): The path to the saved normalizing flow model.

        Returns:
            HybridEmulator: The loaded hybrid emulator model.

        """
        if model_dir is None and (deep_ensemble_path is None or normalizing_flow_path is None):
            raise ValueError("Either model_dir or both deep_ensemble_path and normalizing_flow_path must be provided.")
        if model_dir is not None:
            deep_ensemble_path = f"{model_dir}/deep_ensemble.pth"
            normalizing_flow_path = f"{model_dir}/normalizing_flow.pth"
        deep_ensemble = DeepEnsemble.load(deep_ensemble_path)
        deep_ensemble.trained = True
        normalizing_flow = NormalizingFlow.load(normalizing_flow_path)
        normalizing_flow.trained = True
        model = HybridEmulator(deep_ensemble, normalizing_flow)
        model.trained = True
        return model
