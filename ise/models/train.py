from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from ise.data.dataclasses import PyTorchDataset, TSDataset
from torch.utils.data import DataLoader
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

np.random.seed(10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Class for training neural network emulators. Contains helper functions for handling data as
    well as training, testing, and deploying the neural networks.
    """

    def __init__(
        self,
    ):
        """Initializes class and opens/stores data."""
        self.model = None
        self.data = {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Determine whether on GPU or not
        self.num_input_features = None
        self.loss_logs = {
            "training_loss": [],
            "vasqrtl_loss": [],
        }
        self.train_loader = None
        self.test_loader = None
        self.data_dict = None
        self.logs = {"training": {"epoch": [], "batch": []}, "testing": []}

    def _format_data(
        self,
        train_features: pd.DataFrame,
        train_labels,
        test_features,
        test_labels,
        train_batch_size=100,
        test_batch_size=10,
        sequence_length=5,
    ):
        """Takes training and testing dataframes and converts them into PyTorch DataLoaders to be used in the training loop.

        Args:
            train_features (pd.DataFrame|np.array): training dataset features
            train_labels (pd.Series|np.array): training dataset labels
            test_features (pd.DataFrame|np.array): test dataset features
            test_labels (pd.Series|np.array): test dataset labels
            train_batch_size (int, optional): batch size for training loop. Defaults to 100.
            test_batch_size (int, optional): batch size for validation loop. Defaults to 10.

        Returns:
            self (Trainer): Trainer object
        """

        # Convert to Numpy first (no direct conversion from pd.DataFrame to torch.tensor)
        self.X_train = np.array(train_features, dtype=np.float64)
        self.y_train = np.array(train_labels, dtype=np.float64)
        self.X_test = np.array(test_features, dtype=np.float64)
        self.y_test = np.array(test_labels, dtype=np.float64)

        if self.time_series:
            train_dataset = TSDataset(
                torch.from_numpy(self.X_train).float(),
                torch.from_numpy(self.y_train).float().squeeze(),
                sequence_length=sequence_length,
            )
            test_dataset = TSDataset(
                torch.from_numpy(self.X_test).float(),
                torch.from_numpy(self.y_test).float().squeeze(),
                sequence_length=sequence_length,
            )
        else:
            train_dataset = PyTorchDataset(
                torch.from_numpy(self.X_train).float(),
                torch.from_numpy(self.y_train).float().squeeze(),
            )
            test_dataset = PyTorchDataset(
                torch.from_numpy(self.X_test).float(),
                torch.from_numpy(self.y_test).float().squeeze(),
            )

        # Create dataset and data loaders to be used in training loop
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=train_batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size,
        )

        return self

    def _initiate_model(
        self,
        model_class,
        data_dict,
        architecture,
        sequence_length,
        batch_size,
        mc_dropout,
        dropout_prob,
    ):
        # save attributes
        self.data_dict = data_dict
        self.num_input_features = self.data_dict["train_features"].shape[1]

        # TODO: Write load_saved_model method in model file
        # Loop through possible architecture parameters and if it not given, set it to None
        for param in ["num_linear_layers", "nodes", "num_rnn_hidden", "num_rnn_layers"]:
            try:
                architecture[param]
            except:
                architecture[param] = None
        architecture["input_layer_size"] = self.num_input_features

        # establish model - if using exploratory model, use num_linear_layers and nodes arg
        self.model = model_class(
            architecture=architecture, mc_dropout=mc_dropout, dropout_prob=dropout_prob
        ).to(self.device)
        self.time_series = True if hasattr(self.model, "time_series") else False

        # If the data loader hasn't been created, run _format_data function
        if self.train_loader is None or self.train_loader is None:
            self._format_data(
                data_dict["train_features"],
                data_dict["train_labels"],
                data_dict["test_features"],
                data_dict["test_labels"],
                train_batch_size=batch_size,
                sequence_length=sequence_length,
            )

    def train(
        self,
        model_class,
        data_dict: dict,
        criterion,
        epochs: int,
        batch_size: int,
        mc_dropout: bool = False,
        dropout_prob: float = 0.1,
        tensorboard: bool = False,
        architecture: dict = None,
        save_model: str = False,
        performance_optimized: bool = False,
        verbose: bool = True,
        sequence_length: int = 5,
        tensorboard_comment: str = None,
    ):
        """Training loop for training a PyTorch model. Include validation, GPU compatibility, and tensorboard integration.

        Args:
            model_class (ModelClass): Model to be trained. Usually custom model class.
            data_dict (dict): Dictionary containing training and testing arrays/tensors.
            criterion (torch.nn.Loss): Loss class from PyTorch NN module. Typically torch.nn.MSELoss().
            epochs (int): Number of training epochs
            batch_size (int): Number of training batches
            mc_dropout (bool, optional): Flag denoting whether the model was trained with MC dropout protocol. Defaults to False.
            dropout_prob (float, optional): Dropout probability in MC dropout protocol. Unused if mc_dropout=False. Defaults to 0.1.
            tensorboard (bool, optional): Flag determining whether Tensorboard logs should be generated and outputted. Defaults to False.
            num_linear_layers (int, optional): Number of linear layers in the model. Only used if paired with ExploratoryModel. Defaults to None.
            nodes (list, optional): List of integers denoting the number of nodes in num_linear_layers. Len(nodes) must equal num_linear_layers. Defaults to None.
            save_model (bool, optional): Flag determining whether the trained model should be saved. Defaults to False.
            performance_optimized (bool, optional): Flag determining whether the training loop should be optimized for fast training. Defaults to False.
        
        Returns:
            self (Trainer): Trainer object
        """

        # Initiates model with inputted architecture and formats data
        self._initiate_model(
            model_class,
            data_dict,
            architecture,
            sequence_length,
            batch_size,
            mc_dropout,
            dropout_prob,
        )

        # Use multiple GPU parallelization if available
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

        optimizer = optim.Adam(
            self.model.parameters(),
        )
        # criterion = nn.MSELoss()
        self.time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")

        # tensorboard_comment = f" -- outputs, {self.time}, FC={architecture['num_linear_layers']}, nodes={architecture['nodes']}, batch_size={batch_size},"
        # comment = f" -- {self.time}, dataset={dataset},"
        if tensorboard:
            tb = SummaryWriter(comment=tensorboard_comment)
        mae = nn.L1Loss()

        # Loop through epochs
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_start = time.time()

            total_loss = 0
            total_mae = 0
            
            # for each batch in train_loader
            for X_train_batch, y_train_batch in self.train_loader:
                
                # send to gpu if available
                X_train_batch = X_train_batch.to(self.device)
                y_train_batch = y_train_batch.to(self.device)

                # set gradients to zero for the batch
                optimizer.zero_grad()

                # get prediction and calculate loss
                pred = self.model(X_train_batch)
                loss = criterion(pred, y_train_batch.unsqueeze(1))
                
                # calculate dloss/dx for every parameter x (gradients) and advance optimizer
                loss.backward()
                optimizer.step()

                # add loss to total loss
                total_loss += loss.item()
                self.logs["training"]["batch"].append(loss.item())

                if not performance_optimized:
                    total_mae += mae(pred, y_train_batch.unsqueeze(1)).item()

            # divide total losses by number of batches and save to logs
            avg_mse = total_loss / len(self.train_loader)
            self.logs["training"]["epoch"].append(avg_mse)

            if not performance_optimized:
                avg_rmse = np.sqrt(avg_mse)
                avg_mae = total_mae / len(self.train_loader)

            training_end = time.time()


            # If it isn't performance_optimized, run a validation process as well
            if not performance_optimized:
                self.model.eval()
                test_total_loss = 0
                test_total_mae = 0
                
                # for each batch in the test_loader
                for X_test_batch, y_test_batch in self.test_loader:
                    
                    # send to gpu if available
                    X_test_batch = X_test_batch.to(self.device)
                    y_test_batch = y_test_batch.to(self.device)
                    
                    # get prediction and calculate loss
                    test_pred = self.model(X_test_batch)
                    loss = criterion(test_pred, y_test_batch.unsqueeze(1))
                    
                    # add losses to total epoch loss
                    test_total_loss += loss.item()
                    test_total_mae += mae(test_pred, y_test_batch.unsqueeze(1)).item()

                # divide total losses by number of batches and save to logs
                test_mse = test_total_loss / len(self.test_loader)
                test_mae = test_total_mae / len(self.test_loader)
                self.logs["testing"].append(test_mse)
                testing_end = time.time()

                # get the r2 score for that particular epoch
                preds, means, sd = self.model.predict(self.X_test, mc_iterations=1)
                if self.device.type != "cuda":
                    r2 = r2_score(self.y_test, preds)
                else:
                    r2 = r2_score(self.y_test, preds)

            # if tensorboard, add logs to tensorboard object
            if tensorboard:
                tb.add_scalar("Training MSE", avg_mse, epoch)

                if not performance_optimized:
                    tb.add_scalar("Training RMSE", avg_rmse, epoch)
                    tb.add_scalar("Training MAE", avg_mae, epoch)

                    tb.add_scalar("Validation MSE", test_mse, epoch)
                    tb.add_scalar("Validation RMSE", np.sqrt(test_mse), epoch)
                    tb.add_scalar("Validation MAE", test_mae, epoch)
                    tb.add_scalar("R^2", r2, epoch)


            # if verbose, do all the print statements that are calculated
            if verbose:
                if not performance_optimized:
                    print("")
                    print(
                        f"""Epoch: {epoch}/{epochs}, Training Loss (MSE): {avg_mse:0.8f}, Validation Loss (MSE): {test_mse:0.8f}
    Training time: {training_end - epoch_start: 0.2f} seconds, Validation time: {testing_end - training_end: 0.2f} seconds"""
                    )

                else:
                    print("")
                    print(
                        f"""Epoch: {epoch}/{epochs}, Training Loss (MSE): {avg_mse:0.8f}
    Training time: {training_end - epoch_start: 0.2f} seconds"""
                    )

        if tensorboard:
            metrics, _ = self.evaluate()
            tb.add_hparams(
                {
                    "rnn_layers": architecture["num_rnn_layers"],
                    "hidden": architecture["num_rnn_hidden"],
                    "batch_size": batch_size,
                    "dropout": dropout_prob,
                },
                {
                    "Test MSE": metrics["MSE"],
                    "Test MAE": metrics["MAE"],
                    "R^2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                },
            )

            tb.close()

        # if save_model, save to the state_dict to the desired directory.
        if save_model:
            if isinstance(save_model, str):
                model_path = f"{save_model}/{self.time}, {str(architecture)}.pt"

            elif isinstance(save_model, bool):
                import os

                dirname = os.path.dirname(__file__)
                model_path = os.path.join(dirname, f"../{self.time}, {str(architecture)}.pt")

            torch.save(self.model.state_dict(), model_path)
            print("")
            print(f"Model saved to {model_path}")
            print("")

        return self

    def plot_loss(self, save=False):
        plt.plot(self.logs["training"]["epoch"], "r-", label="Training Loss")
        plt.plot(self.logs["testing"], "b-", label="Validation Loss")
        plt.title("Emulator MSE loss per Epoch")
        plt.xlabel(f"Epoch #")
        plt.ylabel("Loss (MSE)")
        plt.legend()

        if save:
            plt.savefig(save)
        plt.show()

    def evaluate(self, verbose=True):
        # Test predictions
        self.model.eval()
        preds = torch.tensor([]).to(self.device)
        for X_test_batch, y_test_batch in self.test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(self.device), y_test_batch.to(
                self.device
            )
            test_pred = self.model(X_test_batch)
            preds = torch.cat((preds, test_pred), 0)

        if self.device.type == "cuda":
            preds = preds.squeeze().cpu().detach().numpy()
        else:
            preds = preds.squeeze().detach().numpy()

        mse = sum((preds - self.y_test.squeeze()) ** 2) / len(preds)
        mae = sum(abs((preds - self.y_test.squeeze()))) / len(preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, preds)

        metrics = {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}

        if verbose:
            print(
                f"""Test Metrics
MSE: {mse:0.6f}
MAE: {mae:0.6f}
RMSE: {rmse:0.6f}
R2: {r2:0.6f}"""
            )

        return metrics, preds
