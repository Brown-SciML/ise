import torch
from torch import nn, optim
from nflows import distributions, flows, transforms
from ise.utils.functions import to_tensor
from ise.data.ISMIP6.dataclasses import EmulatorDataset
import numpy as np
import json
import os
from ise.utils.training import EarlyStoppingCheckpointer, CheckpointSaver
import wandb

class NormalizingFlow(nn.Module):
    """
    A normalizing flow model for probabilistic modeling using invertible transformations.

    This model utilizes a sequence of invertible transformations to model complex probability 
    distributions. It is built with a base distribution and a series of transformations, 
    leveraging autoregressive neural networks.

    Attributes:
        num_flow_transforms (int): Number of flow transformations in the model.
        num_input_features (int): Number of input features.
        num_predicted_sle (int): Number of predicted sea-level equivalent values.
        flow_hidden_features (int): Number of hidden features in the flow model.
        output_sequence_length (int): Length of the output sequence.
        device (str): Device on which the model is run ("cuda" or "cpu").
        base_distribution (distributions.normal.ConditionalDiagonalNormal): 
            The base normal distribution conditioned on input features.
        t (transforms.base.CompositeTransform): Composite transformation for the normalizing flow.
        flow (flows.base.Flow): The normalizing flow model.
        optimizer (torch.optim.Adam): Optimizer for training the model.
        criterion (callable): Log probability function used as the loss criterion.
        trained (bool): Flag indicating if the model has been trained.
    """

    def __init__(
        self,
        input_size=43,
        output_size=1,
        output_sequence_length=86,
        num_flow_transforms=5,
    ):
        super(NormalizingFlow, self).__init__()
        self.num_flow_transforms = num_flow_transforms
        self.num_input_features = input_size
        self.num_predicted_sle = output_size
        self.flow_hidden_features = output_size * 2
        self.output_sequence_length = output_sequence_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        # Define base distribution
        self.base_distribution = distributions.normal.ConditionalDiagonalNormal(
            shape=[self.num_predicted_sle],
            context_encoder=nn.Linear(self.num_input_features, self.flow_hidden_features),
        )

        # Create flow transforms
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

        # Build flow model
        self.flow = flows.base.Flow(transform=self.t, distribution=self.base_distribution)

        # Define optimizer and criterion
        self.optimizer = optim.Adam(self.flow.parameters())
        self.criterion = self.flow.log_prob
        self.trained = False
        self.wandb_run = None

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=64, save_checkpoints=True, 
            checkpoint_path='checkpoint.pt', early_stopping=True, patience=10, verbose=True, wandb_run=None):
        """
        Trains the normalizing flow model using maximum likelihood estimation.

        Args:
            X (array-like): Input features of shape (num_samples, num_features).
            y (array-like): Target values of shape (num_samples, output_size).
            epochs (int, optional): Number of training epochs. Defaults to 100.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            save_checkpoints (bool, optional): Whether to save model checkpoints. Defaults to True.
            checkpoint_path (str, optional): Path to save model checkpoints. Defaults to 'checkpoint.pt'.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
            patience (int, optional): Number of epochs to wait before early stopping. Defaults to 10.
            verbose (bool, optional): Whether to print training progress. Defaults to True.
            wandb_run (wandb.run, optional): Weights & Biases run for logging. Defaults to None.

        Raises:
            ValueError: If checkpoint loading fails.
        """
        X, y = to_tensor(X).to(self.device), to_tensor(y).to(self.device)
        if y.ndimension() == 1:
            y = y.unsqueeze(1)
        self.wandb_run = wandb_run
        validate = True if X_val is not None and y_val is not None else False
            
        start_epoch = 1
        best_loss = float("inf")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float("inf"))
            if verbose:
                print(f"Resuming from checkpoint at epoch {start_epoch} with validation loss {best_loss:.6f}")
  
        dataset = EmulatorDataset(X, y, sequence_length=1, projection_length=self.output_sequence_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        
        if validate:
            X_val, y_val = to_tensor(X_val).to(self.device), to_tensor(y_val).to(self.device)
            if y_val.ndimension() == 1:
                y_val = y_val.unsqueeze(1)

            val_dataset = EmulatorDataset(X_val, y_val, sequence_length=1, projection_length=self.output_sequence_length)
            val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if save_checkpoints:
            if early_stopping:
                checkpointer = EarlyStoppingCheckpointer(self, self.optimizer, checkpoint_path, patience, verbose)
            else:
                checkpointer = CheckpointSaver(self, self.optimizer, checkpoint_path, verbose)
            checkpointer.best_loss = best_loss

        if start_epoch < epochs:
            for epoch in range(start_epoch, epochs + 1):
                epoch_loss = []
                for i, (x, y) in enumerate(data_loader):
                    x = x.to(self.device).view(x.shape[0], -1)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    loss = torch.mean(-self.flow.log_prob(inputs=y, context=x))
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss.append(loss.item())

                if validate:
                    self.eval()
                    val_losses = []
                    with torch.no_grad():
                        for val_x, val_y in val_data_loader:
                            val_x = val_x.to(self.device).view(val_x.shape[0], -1)
                            val_y = val_y.to(self.device)
                            val_loss = torch.mean(-self.flow.log_prob(inputs=val_y, context=val_x))
                            val_losses.append(val_loss.item())
                    average_epoch_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")

                    train_avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else float("inf")
                    
                    if self.wandb_run:
                        log_dict = {"epoch": epoch, "val_loss": average_epoch_loss}
                        if train_avg_loss is not None:
                            log_dict["train_loss"] = train_avg_loss
                        self.wandb_run.log(log_dict)
                    self.train()
                else:
                    average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
                    if self.wandb_run:
                        self.wandb_run.log({"epoch": epoch, "loss": average_epoch_loss})
                        
                if save_checkpoints:
                    checkpointer(average_epoch_loss, epoch)
                    if hasattr(checkpointer, "early_stop") and checkpointer.early_stop:
                        if self.wandb_run:
                            artifact = wandb.Artifact("nf-model", type='model')
                            artifact.add_file(checkpoint_path)
                            self.wandb_run.log_artifact(artifact)
                        if verbose:
                            print("Early stopping")
                        break

                if verbose:
                    print(f"[epoch/total]: [{epoch}/{epochs}], loss: {average_epoch_loss}{f' -- {checkpointer.log}' if save_checkpoints else ''}")
        else:
            if verbose:
                print(f"Training already completed ({epochs}/{epochs}).")
                    
        self.trained = True
        
        if save_checkpoints:
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint.keys():
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.best_loss = checkpoint['best_loss']
                self.epochs_trained = checkpoint['epoch']
            else:
                self.load_state_dict(checkpoint)
            os.remove(checkpoint_path)

    def sample(self, features, num_samples, return_type="numpy"):
        """
        Generates samples from the trained normalizing flow model.

        Args:
            features (array-like or torch.Tensor): Input features to condition the samples on.
            num_samples (int): Number of samples to generate per input feature set.
            return_type (str, optional): Return type, either "numpy" or "tensor". Defaults to "numpy".

        Returns:
            np.ndarray or torch.Tensor: Generated samples of shape (num_samples, output_size).
        """

        features = to_tensor(features)
        samples = self.flow.sample(num_samples, context=features).reshape(features.shape[0], num_samples)
        if return_type == "numpy":
            return samples.detach().cpu().numpy()
        return samples

    def get_latent(self, x, latent_constant=0.0):
        """
        Computes the latent space representation of the given input.

        Args:
            x (array-like or torch.Tensor): Input data of shape (num_samples, num_features).
            latent_constant (float, optional): Constant value used for latent variable sampling. Defaults to 0.0.

        Returns:
            torch.Tensor: Latent space representation of the input data.
        """

        x = to_tensor(x).to(self.device)
        latent_constant_tensor = torch.ones((x.shape[0], 1)).to(self.device) * latent_constant
        z, _ = self.t(latent_constant_tensor.float(), context=x)
        return z

    def aleatoric(self, features, num_samples, batch_size=128):
        """
        Estimates aleatoric uncertainty by computing the standard deviation of multiple 
        samples drawn from the normalizing flow model.

        Args:
            features (array-like or torch.Tensor): Input features for sampling.
            num_samples (int): Number of samples per input feature set.
            batch_size (int, optional): Batch size for processing features. Defaults to 128.

        Returns:
            np.ndarray: Aleatoric uncertainty estimates of shape (num_samples,).
        """

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
        Saves the trained model and its metadata.

        Args:
            path (str): Path to save the model checkpoint.

        Raises:
            ValueError: If the model has not been trained before saving.
        """

        if not self.trained:
            raise ValueError("Train the model before saving.")
        metadata = {
            "input_size": self.num_input_features,
            "output_size": self.num_predicted_sle,
            "device": self.device,
            "best_loss": self.best_loss,
            "epochs_trained": self.epochs_trained,
        }
        metadata_path = path + "_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'trained': self.trained,
        }, path)
        print(f"Model and metadata saved to {path} and {metadata_path}, respectively.")

    @staticmethod
    def load(path):
        """
        Loads a trained normalizing flow model from a saved checkpoint.

        Args:
            path (str): Path to the saved model checkpoint.

        Returns:
            NormalizingFlow: A restored instance of the NormalizingFlow model.
        """

        metadata_path = path + "_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        model = NormalizingFlow(
            input_size=metadata["input_size"], 
            output_size=metadata["output_size"]
        )

        checkpoint = torch.load(path, map_location="cpu" if not torch.cuda.is_available() else None)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.trained = checkpoint['trained']
        else:
            model.load_state_dict(checkpoint)
            model.trained = True
            
        model.trained = True
        model.to(model.device)
        model.eval()
        return model
