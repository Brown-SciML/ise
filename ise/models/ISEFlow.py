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
from torch.nn import functional as F
from nflows import distributions, flows, transforms
from torch import nn, optim

from ise.data.dataclasses import EmulatorDataset
from ise.data.scaler import LogScaler, RobustScaler, StandardScaler
from ise.models.loss import MSEDeviationLoss, WeightedMSELoss
from ise.utils.functions import to_tensor
from ise.utils.training import EarlyStoppingCheckpointer, CheckpointSaver
from ise.evaluation import metrics as m
from ise.models.predictors.lstm import LSTM




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
        fit(X, y, epochs=100, batch_size=64, patience=10, delta=0, early_stopping_path='checkpoint.pt'): Trains the model on the given input and output data.
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
        num_flow_transforms=5,
    ):
        super(NormalizingFlow, self).__init__()
        self.num_flow_transforms = num_flow_transforms
        self.num_input_features = forcing_size
        self.num_predicted_sle = sle_size
        self.flow_hidden_features = sle_size * 2
        self.projection_length = projection_length
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

    def fit(self, X, y, epochs=100, batch_size=64, save_checkpoints=True, checkpoint_path='checkpoint.pt', early_stopping=True, patience=10, delta=1e-3, verbose=True):
        """
        Trains the model on the given input and output data with early stopping.

        Args:
            X (array-like): The input data.
            y (array-like): The output data.
            epochs (int): The number of training epochs (default: 100).
            batch_size (int): The batch size for training (default: 64).
            patience (int, optional): The number of epochs to wait for improvement before stopping. Defaults to 10.
            delta (float, optional): The minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
            early_stopping_path (str, optional): The path to save the model with the best training loss. Defaults to 'checkpoint.pt'.
        """
        X, y = to_tensor(X).to(self.device), to_tensor(y).to(self.device)
        if y.ndimension() == 1:
            y = y.unsqueeze(1)
        dataset = EmulatorDataset(X, y, sequence_length=1, projection_length=self.projection_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()

        # Initialize early stopping
        if save_checkpoints:
            if early_stopping:
                checkpointer = EarlyStoppingCheckpointer(self, checkpoint_path, patience, verbose)
            else:
                checkpointer = CheckpointSaver(self, checkpoint_path, verbose)

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
            average_epoch_loss = sum(epoch_loss) / len(epoch_loss)

            # Check early stopping
            if early_stopping:
                checkpointer(average_epoch_loss, self)

                if checkpointer.early_stop:
                    print("Early stopping")
                    break
            if verbose:
                print(f"[epoch/total]: [{epoch}/{epochs}], loss: {average_epoch_loss}{f' -- {checkpointer.log}' if early_stopping else ''}")
            

        self.trained = True
        # Load the best model checkpoint
        if early_stopping:
            self.load_state_dict(torch.load(checkpoint_path))
            os.remove(checkpoint_path)

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

    def aleatoric(self, features, num_samples, batch_size=128):
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
            
        num_batches = (features.shape[0] + batch_size - 1) // batch_size
        aleatoric_uncertainty = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i+1)*batch_size, features.shape[0])
            batch_features = features[start_idx:end_idx]
            
            samples = self.flow.sample(num_samples, context=batch_features)
            samples = samples.detach().cpu().numpy()
            std = np.std(samples, axis=1).squeeze()
            aleatoric_uncertainty.append(std)
            
        return np.concatenate(aleatoric_uncertainty)

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


class ISEFlow(torch.nn.Module):
    """
    The ISEFlow (Flow-based Ice Sheet Emulator) that combines a deep ensemble and a normalizing flow model.

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
        super(ISEFlow, self).__init__()

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

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=None, epochs=100, nf_epochs=None, 
            de_epochs=None, sequence_length=5, patience=10, delta=0, 
            early_stopping_path='checkpoint_ensemble', verbose=True):
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
        
        # Handling early stopping (if validation data is provided, turn it on and send a notification)
        if early_stopping is None:
            if X_val is not None and y_val is None:
                early_stopping = True
                print('Validation data provided and early_stopping argument is None, early stopping enabled.')
            else:
                early_stopping = False
            
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
            print(f"\nTraining Normalizing Flow ({'Maximum ' if early_stopping else ''}{nf_epochs} epochs):")
            self.normalizing_flow.fit(X, y, early_stopping=early_stopping, patience=patience, 
                                      delta=delta, epochs=nf_epochs, verbose=verbose, checkpoint_path=early_stopping_path)
        z = self.normalizing_flow.get_latent(
            X,
        ).detach()
        X_latent = torch.concatenate((X, z), axis=1)
        
        X_val_latent = None
        if X_val is not None and y_val is not None:
            X_val, y_val = to_tensor(X_val).to(self.device), to_tensor(y_val).to(self.device)
            z = self.normalizing_flow.get_latent(X_val,).detach()
            X_val_latent = torch.concatenate((X_val, z), axis=1)
        
        if not self.deep_ensemble.trained:
            print(f"\nTraining Deep Ensemble ({'Maximum ' if early_stopping else ''}{de_epochs} epochs):")
            self.deep_ensemble.fit(
                X_latent, y, X_val=X_val_latent, y_val=y_val, early_stopping=early_stopping, 
                patience=patience, delta=delta, early_stopping_path=early_stopping_path,
                epochs=de_epochs, sequence_length=sequence_length, verbose=verbose,
                )
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

        import time
        start_time = time.time()
        predictions, uncertainties = self.forward(x, smooth_projection=smooth_projection)
        print('forward time:', time.time() - start_time)
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

    def save(self, save_dir, input_features=None):
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
        
        if input_features is not None:
            if not isinstance(input_features, list):
                raise ValueError("input_features must be a list of feature names")
            else:
                with open(f"{save_dir}/input_features.json", "w") as f:
                    json.dump(input_features, f, indent=4)

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
        model = ISEFlow(deep_ensemble, normalizing_flow)
        model.trained = True
        return model


