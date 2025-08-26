import torch
from torch import nn, optim
import torch.nn.functional as F
import warnings
import pandas as pd
import numpy as np
import os

from ise.utils.functions import to_tensor
from ise.data.ISMIP6.dataclasses import EmulatorDataset
from ise.utils.training import CheckpointSaver, EarlyStoppingCheckpointer

class LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) model for time series forecasting.

    This class implements an LSTM network with multiple layers, dropout, and fully connected
    layers to generate predictions for sequential data.

    Attributes:
        lstm_num_layers (int): Number of LSTM layers in the model.
        lstm_num_hidden (int): Number of hidden units in each LSTM layer.
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        output_sequence_length (int): Number of time steps predicted by the model.
        device (str): Device on which the model runs ('cuda' or 'cpu').
        lstm (nn.LSTM): LSTM layer for sequence modeling.
        relu (nn.ReLU): ReLU activation function.
        linear1 (nn.Linear): Intermediate fully connected layer.
        linear_out (nn.Linear): Output layer mapping to final predictions.
        optimizer (torch.optim.Optimizer): Optimization algorithm used for training.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
        criterion (torch.nn.modules.loss._Loss): Loss function used for training.
        trained (bool): Flag indicating whether the model has been trained.

    Args:
        lstm_num_layers (int): Number of LSTM layers.
        lstm_hidden_size (int): Number of hidden units in each LSTM layer.
        input_size (int, optional): Number of input features. Defaults to 83.
        output_size (int, optional): Number of output features. Defaults to 1.
        criterion (torch.nn.modules.loss._Loss, optional): Loss function. Defaults to MSELoss.
        output_sequence_length (int, optional): Number of output time steps. Defaults to 86.
        optimizer (torch.optim.Optimizer, optional): Optimizer type. Defaults to Adam.
    """

    def __init__(
        self,
        lstm_num_layers,
        lstm_hidden_size,
        input_size=83,
        output_size=1,
        criterion=torch.nn.MSELoss(),
        output_sequence_length=86,
        optimizer=optim.Adam
    ):
        super(LSTM, self).__init__()

        # Initialize attributes
        self.lstm_num_layers = int(lstm_num_layers)
        self.lstm_num_hidden = int(lstm_hidden_size)
        self.input_size = input_size
        self.output_size = output_size
        self.output_sequence_length = output_sequence_length
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
        self.optimizer = optimizer(self.parameters())
        self.dropout = nn.Dropout(p=0.2)
        self.criterion = criterion
        self.trained = False

    def forward(self, x):
        """
        Performs a forward pass through the LSTM network.

        Given an input sequence, the LSTM processes the sequence to extract features,
        which are passed through a fully connected network to generate predictions.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size), representing 
            the model’s predictions.
        """

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
        x = self.linear_out(x)

        return x
    
    
    def fit(self, X, y, epochs=100, sequence_length=5, batch_size=64, criterion=None, X_val=None, y_val=None, 
            save_checkpoints=True, checkpoint_path='checkpoint.pt', early_stopping=False, patience=10, 
            verbose=True, dataclass=EmulatorDataset):
        """
        Trains the LSTM model on the provided data.

        Supports optional checkpointing and early stopping. If a checkpoint exists, 
        training resumes from the last saved state.

        Args:
            X (Tensor or DataFrame): Input training data.
            y (Tensor or DataFrame): Target values corresponding to the input data.
            epochs (int, optional): Number of epochs for training. Defaults to 100.
            sequence_length (int, optional): Length of input sequences. Defaults to 5.
            batch_size (int, optional): Batch size used in training. Defaults to 64.
            criterion (torch.nn.modules.loss._Loss, optional): Loss function. Defaults to None.
            X_val (Tensor or DataFrame, optional): Validation input data. Defaults to None.
            y_val (Tensor or DataFrame, optional): Validation target data. Defaults to None.
            save_checkpoints (bool, optional): Whether to save model checkpoints. Defaults to True.
            checkpoint_path (str, optional): Path to save model checkpoints. Defaults to 'checkpoint.pt'.
            early_stopping (bool, optional): Whether to enable early stopping. Defaults to False.
            patience (int, optional): Number of epochs to wait before stopping. Defaults to 10.
            verbose (bool, optional): Whether to print training progress. Defaults to True.
            dataclass (type, optional): Dataset class for handling data. Defaults to EmulatorDataset.

        Raises:
            ValueError: If no loss function is provided.

        Notes:
            - If validation data is provided but early stopping is disabled, a warning is issued.
            - If a checkpoint exists, training resumes from the saved epoch.
            - If early stopping is enabled, the model stops training when validation loss stops improving.
        """

        X, y = to_tensor(X).to(self.device), to_tensor(y).to(self.device)
        if y.ndimension() == 1:
            y = y.unsqueeze(1)
            
        # Check if a checkpoint exists and load it
        start_epoch = 1
        best_loss = float("inf")
        self.checkpoint_path = checkpoint_path
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float("inf"))
            if verbose:
                print(f"Resuming from checkpoint at epoch {start_epoch} with validation loss {best_loss:.6f}")

        
        # Check if validation data is provided
        if X_val is not None and y_val is not None:
            validate = True
            if not early_stopping:
                warnings.warn(
                    "Validation data provided but early_stopping is False. Early stopping is recommended for validation data."
                )
            X_val, y_val = to_tensor(X_val).to(self.device), to_tensor(y_val).to(self.device)
        else:
            validate = False

        # Set loss criterion
        if criterion is not None:
            self.criterion = criterion.to(self.device)
        elif criterion is None and self.criterion is None:
            raise ValueError("loss must be provided if criterion is None.")
        self.criterion = self.criterion.to(self.device)

        # Convert data to numpy arrays if pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        # Create dataset and data loader
        dataset = dataclass(X, y, sequence_length=sequence_length, projection_length=self.output_sequence_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set model to training mode
        self.train()
        self.to(self.device)

        # Initialize early stopping
        if save_checkpoints:
            if early_stopping:
                checkpointer = EarlyStoppingCheckpointer(self, self.optimizer, checkpoint_path, patience, verbose)
            else:
                checkpointer = CheckpointSaver(self, self.optimizer, checkpoint_path, verbose)
                
            checkpointer.best_loss = best_loss
        else:
            checkpointer = None
        
        # Training loop
        if start_epoch < epochs:
            for epoch in range(start_epoch, epochs + 1):
                self.train()
                batch_losses = []
                for i, (x, y) in enumerate(data_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    y_pred = self.forward(x)
                    loss = self.criterion(y_pred, y)  # Renamed to 'loss' for clarity
                    loss.backward()
                    self.optimizer.step()
                    batch_losses.append(loss.item())

                # Print average batch loss and validation loss (if provided)
                if validate:
                    val_preds = self.predict(
                        X_val, sequence_length=sequence_length, batch_size=batch_size
                    ).to(self.device)
                    val_loss = F.mse_loss(val_preds.squeeze(), y_val.squeeze())

                    if save_checkpoints:
                        checkpointer(val_loss, epoch)

                        if hasattr(checkpointer, "early_stop") and checkpointer.early_stop:
                            if verbose:
                                print("Early stopping") 
                            break
                        
                    if verbose:
                        print(f"[epoch/total]: [{epoch}/{epochs}], train loss: {sum(batch_losses) / len(batch_losses)}, val mse: {val_loss:.6f} -- {getattr(checkpointer, 'log', '') if checkpointer is not None else ''}")
                else:
                    average_batch_loss = sum(batch_losses) / len(batch_losses)
                    if verbose:
                        print(f"[epoch/total]: [{epoch}/{epochs}], train loss: {average_batch_loss}")
        else:
            if verbose:
                print(f"Training already completed ({epochs}/{epochs}).")

        self.trained = True
        
        # loads best model
        if save_checkpoints:
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint.keys():
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.best_loss = checkpoint['best_loss']
                self.epochs_trained = checkpoint['epoch']
            else:
                self.load_state_dict(checkpoint)
            # os.remove(checkpoint_path)

    def predict(self, X, sequence_length=5, batch_size=64, dataclass=EmulatorDataset):
        """
        Generates predictions using the trained LSTM model.

        The model processes input sequences and returns predictions. Predictions are computed
        in a batch-wise manner to optimize memory usage.

        Args:
            X (Tensor or DataFrame): Input data for prediction.
            sequence_length (int, optional): Length of input sequences. Defaults to 5.
            batch_size (int, optional): Batch size used for inference. Defaults to 64.
            dataclass (type, optional): Dataset class for handling data. Defaults to EmulatorDataset.

        Returns:
            Tensor: Predicted values for the input data.

        Notes:
            - The model is set to evaluation mode before making predictions.
            - Data is converted to tensors if initially provided as pandas DataFrames.
        """

        self.eval()
        self.to(self.device)

        # Convert data to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Create dataset and data loader
        dataset = dataclass(X, y=None, sequence_length=sequence_length, projection_length=self.output_sequence_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = torch.tensor([]).to(self.device)
        for X_test_batch in data_loader:
            self.eval()
            X_test_batch = X_test_batch.to(self.device)
            y_pred = self.forward(X_test_batch)
            preds = torch.cat((preds, y_pred), 0)

        return preds
