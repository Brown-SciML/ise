import json
import os
import pickle
import warnings

import numpy as np
import torch
from torch import nn, optim
from nflows import distributions, flows, transforms

from ise.data.dataclasses import EmulatorDataset
from ise.utils.functions import to_tensor
from ise.utils.training import EarlyStoppingCheckpointer, CheckpointSaver
from ise.models.predictors.deep_ensemble import DeepEnsemble
from ise.models.density_estimators.normalizing_flow import NormalizingFlow
from .de import ISEFlow_AIS_DE, ISEFlow_GrIS_DE
from .nf import ISEFlow_AIS_NF, ISEFlow_GrIS_NF

class ISEFlow(torch.nn.Module):
    """
    The ISEFlow (Flow-based Ice Sheet Emulator) that combines a deep ensemble and a normalizing flow model.
    """

    def __init__(self, deep_ensemble, normalizing_flow):
        super(ISEFlow, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        if not isinstance(deep_ensemble, DeepEnsemble):
            raise ValueError("deep_ensemble must be a DeepEnsemble instance")
        if not isinstance(normalizing_flow, NormalizingFlow):
            raise ValueError("normalizing_flow must be a NormalizingFlow instance")

        self.deep_ensemble = deep_ensemble.to(self.device)
        self.normalizing_flow = normalizing_flow.to(self.device)
        self.trained = self.deep_ensemble.trained and self.normalizing_flow.trained
        self.scaler_path = None

    def fit(self, X, y, nf_epochs, de_epochs, batch_size=64, X_val=None, y_val=None, save_checkpoints=True, checkpoint_path='checkpoint_ensemble', early_stopping=True,  
             sequence_length=5, patience=10, verbose=True):
        """
        Fits the hybrid emulator to the training data.
        """
        
        if early_stopping is None:
            early_stopping = X_val is not None and y_val is not None

        torch.manual_seed(np.random.randint(0, 100000))

        X, y = to_tensor(X).to(self.device), to_tensor(y).to(self.device)

        if self.trained:
            warnings.warn("This model has already been trained. Training again.")

        # Train Normalizing Flow
        if not self.normalizing_flow.trained:
            print(f"\nTraining Normalizing Flow ({'Maximum ' if early_stopping else ''}{nf_epochs} epochs):")
            self.normalizing_flow.fit(X, y, nf_epochs, batch_size, save_checkpoints, f"{checkpoint_path}_nf.pth", early_stopping, patience, verbose)      

        # Latent representation
        z = self.normalizing_flow.get_latent(X).detach()
        X_latent = torch.cat((X, z), axis=1)
        
        X_val_latent = None
        if X_val is not None and y_val is not None:
            X_val, y_val = to_tensor(X_val).to(self.device), to_tensor(y_val).to(self.device)
            z_val = self.normalizing_flow.get_latent(X_val).detach()
            X_val_latent = torch.cat((X_val, z_val), axis=1)
        
        # Train Deep Ensemble
        if not self.deep_ensemble.trained:
            print(f"\nTraining Deep Ensemble ({'Maximum ' if early_stopping else ''}{de_epochs} epochs):")
            
            self.deep_ensemble.fit(X_latent, y, X_val_latent, y_val, save_checkpoints, f"{checkpoint_path}_de", early_stopping, de_epochs, batch_size, sequence_length, patience, verbose,)

        self.trained = True

    def forward(self, x, smooth_projection=False):
        """
        Performs a forward pass through the hybrid emulator.
        """
        self.eval()
        x = to_tensor(x).to(self.device)
        if not self.trained:
            warnings.warn("This model has not been trained. Predictions may not be accurate.")
        z = self.normalizing_flow.get_latent(x).detach()
        X_latent = torch.cat((x, z), axis=1)
        prediction, epistemic = self.deep_ensemble(X_latent)
        aleatoric = self.normalizing_flow.aleatoric(x, 100)
        prediction = prediction.detach().cpu().numpy()
        epistemic = epistemic.detach().cpu().numpy()
        uncertainties = dict(
            total=aleatoric + epistemic,
            epistemic=epistemic,
            aleatoric=aleatoric,
        )
        return prediction, uncertainties

    def predict(self, x, output_scaler=None, smooth_projection=False):
        self.eval()
        if output_scaler is None and self.scaler_path is None:
            warnings.warn("No scaler path provided, uncertainties are not in units of SLE.")
            return self.forward(x, smooth_projection=smooth_projection)

        if isinstance(output_scaler, str):
            self.scaler_path = output_scaler
            with open(self.scaler_path, "rb") as f:
                output_scaler = pickle.load(f)

        predictions, uncertainties = self.forward(x, smooth_projection=smooth_projection)
        unscaled_predictions = output_scaler.inverse_transform(predictions.reshape(-1, 1))
        bound_epistemic = predictions + uncertainties["epistemic"]
        bound_aleatoric = predictions + uncertainties["aleatoric"]
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
        """
        if not self.trained:
            raise ValueError("This model has not been trained yet. Train the model before saving.")
        if save_dir.endswith(".pth"):
            raise ValueError("save_dir must be a directory, not a file")
        os.makedirs(save_dir, exist_ok=True)

        self.deep_ensemble.save(os.path.join(save_dir, "deep_ensemble.pth"))
        self.normalizing_flow.save(os.path.join(save_dir, "normalizing_flow.pth"))
        
        if input_features is not None:
            if not isinstance(input_features, list):
                raise ValueError("input_features must be a list of feature names")
            with open(os.path.join(save_dir, "input_features.json"), "w") as f:
                json.dump(input_features, f, indent=4)

    @staticmethod
    def load(model_dir=None, deep_ensemble_path=None, normalizing_flow_path=None):
        """
        Loads a trained model from the specified paths.
        """
        if model_dir:
            deep_ensemble_path = os.path.join(model_dir, "deep_ensemble.pth")
            normalizing_flow_path = os.path.join(model_dir, "normalizing_flow.pth")
        
        deep_ensemble = DeepEnsemble.load(deep_ensemble_path)
        normalizing_flow = NormalizingFlow.load(normalizing_flow_path)
        model = ISEFlow(deep_ensemble, normalizing_flow)
        model.trained = True
        return model


class ISEFlow_AIS(ISEFlow):
    def __init__(self,):
        deep_ensemble = ISEFlow_AIS_DE()
        normalizing_flow = ISEFlow_AIS_NF()
        super(ISEFlow_AIS, self).__init__(deep_ensemble, normalizing_flow)
        

class ISEFlow_GrIS(ISEFlow):
    def __init__(self,):
        deep_ensemble = ISEFlow_GrIS_DE()
        normalizing_flow = ISEFlow_GrIS_NF()
        super(ISEFlow_GrIS, self).__init__(deep_ensemble, normalizing_flow)