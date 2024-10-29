import torch
from torch import nn, optim
from nflows import distributions, flows, transforms
from ise.utils.functions import to_tensor
from ise.data.dataclasses import EmulatorDataset
import numpy as np
import json
import os
from ise.utils.training import EarlyStoppingCheckpointer, CheckpointSaver

class NormalizingFlow(nn.Module):
    """
    A class representing a Normalizing Flow model.
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

    def fit(self, X, y, epochs=100, batch_size=64, save_checkpoints=True, checkpoint_path='checkpoint.pt', early_stopping=True, patience=10, verbose=True):
        X, y = to_tensor(X).to(self.device), to_tensor(y).to(self.device)
        if y.ndimension() == 1:
            y = y.unsqueeze(1)
            
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

        if save_checkpoints:
            if early_stopping:
                checkpointer = EarlyStoppingCheckpointer(self, self.optimizer, checkpoint_path, patience, verbose)
            else:
                checkpointer = CheckpointSaver(self, self.optimizer, checkpoint_path, verbose)
            checkpointer.best_loss = best_loss

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
            average_epoch_loss = sum(epoch_loss) / len(epoch_loss)

            if save_checkpoints:
                checkpointer(average_epoch_loss)
                if hasattr(checkpointer, "early_stop") and checkpointer.early_stop:
                    if verbose:
                        print("Early stopping")
                    break

            if verbose:
                print(f"[epoch/total]: [{epoch}/{epochs}], loss: {average_epoch_loss}{f' -- {checkpointer.log}' if early_stopping else ''}")
            
        self.trained = True
        
        if early_stopping:
            self.load_state_dict(torch.load(checkpoint_path))
            os.remove(checkpoint_path)

    def sample(self, features, num_samples, return_type="numpy"):
        features = to_tensor(features)
        samples = self.flow.sample(num_samples, context=features).reshape(features.shape[0], num_samples)
        if return_type == "numpy":
            return samples.detach().cpu().numpy()
        return samples

    def get_latent(self, x, latent_constant=0.0):
        x = to_tensor(x).to(self.device)
        latent_constant_tensor = torch.ones((x.shape[0], 1)).to(self.device) * latent_constant
        z, _ = self.t(latent_constant_tensor.float(), context=x)
        return z

    def aleatoric(self, features, num_samples, batch_size=128):
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
        if not self.trained:
            raise ValueError("Train the model before saving.")
        metadata = {
            "input_size": self.num_input_features,
            "output_size": self.num_predicted_sle,
            "device": self.device
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
        metadata_path = path + "_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        model = NormalizingFlow(
            input_size=metadata["input_size"], 
            output_size=metadata["output_size"]
        )

        checkpoint = torch.load(path, map_location="cpu" if not torch.cuda.is_available() else None)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.trained = checkpoint['trained']
        model.to(model.device)
        model.eval()
        return model
