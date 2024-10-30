import torch
from torch import nn
import numpy as np
import warnings
import os
import json

from ise.models.predictors.lstm import LSTM

class DeepEnsemble(nn.Module):

    def __init__(self, ensemble_members=None, input_size=83, output_size=1, num_ensemble_members=3, output_sequence_length=86, latent_dim=1):
        super(DeepEnsemble, self).__init__()
        self.input_size = input_size + latent_dim
        self.output_size = output_size
        self.output_sequence_length = output_sequence_length
        self.loss_choices = [torch.nn.MSELoss(), torch.nn.L1Loss(), torch.nn.HuberLoss()]

        # Initialize ensemble members
        if not ensemble_members:
            self.ensemble_members = [
                LSTM(
                    lstm_num_layers=np.random.randint(1, 3),
                    lstm_hidden_size=np.random.choice([512, 256, 128, 64]),
                    criterion=np.random.choice(self.loss_choices),
                    input_size=self.input_size,
                    output_size=self.output_size,
                    output_sequence_length=self.output_sequence_length,
                )
                for _ in range(num_ensemble_members)
            ]
        elif isinstance(ensemble_members, list) and all(isinstance(m, LSTM) for m in ensemble_members):
            self.ensemble_members = ensemble_members
        else:
            raise ValueError("ensemble_members must be a list of LSTM instances")

        # Check if all ensemble members are trained
        self.trained = all([member.trained for member in self.ensemble_members])

    def forward(self, x):
        """
        Performs a forward pass through the ensemble.

        Args:
        - x: Input data.

        Returns:
        - mean_prediction: Mean prediction across ensemble members.
        - epistemic_uncertainty: Standard deviation across ensemble predictions.
        """
        if not self.trained:
            warnings.warn("This model has not been trained. Predictions may be inaccurate.")
        preds = torch.cat([member.predict(x).unsqueeze(1) for member in self.ensemble_members], dim=1)
        mean_prediction = preds.mean(dim=1).squeeze()
        epistemic_uncertainty = preds.std(dim=1).squeeze()
        return mean_prediction, epistemic_uncertainty

    def predict(self, x):
        """
        Makes predictions using the ensemble.

        Args:
        - x: Input data.

        Returns:
        - Tuple[Tensor, Tensor]: Mean predictions and uncertainty estimates.
        """
        self.eval()
        return self.forward(x)

    def fit(self, X, y, X_val=None, y_val=None, save_checkpoints=True, checkpoint_path='checkpoint_ensemble', early_stopping=False, epochs=100, batch_size=128, sequence_length=5, patience=10, verbose=True):
        """
        Trains the ensemble with optional early stopping.

        Args:
        - X, y: Training data.
        - early_stopping (bool): Use early stopping. Defaults to False.
        """
        if self.trained:
            warnings.warn("Model already trained. Proceeding to train again.")
        for i, member in enumerate(self.ensemble_members):
            if verbose:
                print(f"Training Ensemble Member {i+1} of {len(self.ensemble_members)}:")
            member.fit(X, y, X_val=X_val, y_val=y_val, epochs=epochs, batch_size=batch_size, sequence_length=sequence_length, save_checkpoints=save_checkpoints, checkpoint_path=f'{checkpoint_path}_member{i+1}.pth', early_stopping=early_stopping, patience=patience, verbose=verbose)
            print("")
        self.trained = True

    def save(self, model_path):
        if not self.trained:
            raise ValueError("Train the model before saving.")
        
        # Ensure the save directory is based on model_path
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        ensemble_dir = os.path.join(model_dir, "ensemble_members")
        os.makedirs(ensemble_dir, exist_ok=True)

        # Prepare metadata for each ensemble member with paths relative to the model directory
        metadata = {
            "model_type": self.__class__.__name__,
            "version": "1.0",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "ensemble_members": [
                {
                    "lstm_num_layers": member.lstm_num_layers,
                    "lstm_num_hidden": member.lstm_num_hidden,
                    "criterion": member.criterion.__class__.__name__,
                    "input_size": member.input_size,
                    "output_size": member.output_size,
                    "trained": member.trained,
                    "path": os.path.join("ensemble_members", f"member_{i+1}.pth"),
                }
                for i, member in enumerate(self.ensemble_members)
            ],
        }

        # Save metadata file in the same directory as the model
        metadata_path = model_path.replace(".pth", "_metadata.json")
        with open(metadata_path, "w") as file:
            json.dump(metadata, file, indent=4)
        print(f"Model metadata saved to {metadata_path}")

        # Save the state dictionary of the ensemble model
        torch.save(self.state_dict(), model_path)
        print(f"Model parameters saved to {model_path}")

        # Save each ensemble member’s state dict in the ensemble directory
        for i, member in enumerate(self.ensemble_members):
            member_path = os.path.join(ensemble_dir, f"member_{i+1}.pth")
            torch.save(member.state_dict(), member_path)
            print(f"Ensemble Member {i+1} saved to {member_path}")

    @classmethod
    def load(cls, model_path):
        metadata_path = model_path.replace(".pth", "_metadata.json")
        model_dir = os.path.dirname(model_path)

        with open(metadata_path, "r") as file:
            metadata = json.load(file)

        if cls.__name__ != metadata["model_type"]:
            raise ValueError(f"Metadata type {metadata['model_type']} does not match {cls.__name__}")

        loss_lookup = {"MSELoss": torch.nn.MSELoss(), "L1Loss": torch.nn.L1Loss(), "HuberLoss": torch.nn.HuberLoss()}
        ensemble_members = []

        # Load each ensemble member from the same directory
        for member_metadata in metadata["ensemble_members"]:
            member_path = os.path.join(model_dir, member_metadata["path"])
            if not os.path.isfile(member_path):
                raise FileNotFoundError(f"Ensemble member file not found: {member_path}")
            
            criterion = loss_lookup[member_metadata["criterion"]]
            member = LSTM(
                lstm_num_layers=member_metadata["lstm_num_layers"],
                lstm_hidden_size=member_metadata["lstm_num_hidden"],
                input_size=member_metadata["input_size"],
                output_size=member_metadata["output_size"],
                criterion=criterion,
            )
            state_dict = torch.load(member_path, map_location="cpu" if not torch.cuda.is_available() else None)
            member.load_state_dict(state_dict)
            member.trained = True
            member.eval()
            ensemble_members.append(member)

        model = cls(ensemble_members=ensemble_members)
        ensemble_state_dict = torch.load(model_path, map_location="cpu" if not torch.cuda.is_available() else None)
        model.load_state_dict(ensemble_state_dict, strict=False)
        model.eval()
        return model
