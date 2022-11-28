from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from training.PyTorchDataset import PyTorchDataset, TSDataset
from torch.utils.data import DataLoader
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, cfg):
        self.model = None
        self.data = {}
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determine whether on GPU or not
        # self.data_dir = self.cfg['data']['directory'] if self.cfg['data']['directory'] is not None else self.cfg['data']['export']
        # self.model_name = cfg['training']['model']
        # self.num_epochs = cfg['training']['epochs']
        # self.verbose = cfg['training']['verbose']
        # self.batch_size = cfg['training']['batch_size']
        self.num_input_features = None
        self.loss_logs = {'training_loss': [], 'vasqrtl_loss': [], }
        self.train_loader = None
        self.test_loader = None
        self.data_dict = None
        self.logs = {'training': {'epoch': [], 'batch': []}, 'testing': []}

    def _format_data(self, train_features, train_labels, test_features, test_labels, train_batch_size=100,
                     test_batch_size=10, ):
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
                sequence_length=3,
            )
            test_dataset = TSDataset(
                torch.from_numpy(self.X_test).float(),
                torch.from_numpy(self.y_test).float().squeeze(),
                sequence_length=3,
            )
        else:
            train_dataset = PyTorchDataset(torch.from_numpy(self.X_train).float(),
                                           torch.from_numpy(self.y_train).float().squeeze())
            test_dataset = PyTorchDataset(torch.from_numpy(self.X_test).float(),
                                          torch.from_numpy(self.y_test).float().squeeze())

        # Create dataset and data loaders to be used in training loop
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, )

        return self

    def train(self, model, data_dict, criterion, epochs, batch_size, tensorboard=False, architecture=None,
              save_model=False, performance_optimized=False):
        """Training loop for training a PyTorch model. Include validation, GPU compatibility, and tensorboard integration.

        Args:
            model (ModelClass): Model to be trained. Usually custom model class.
            data_dict (dict): Dictionary containing training and testing arrays/tensors.
                    Example: {'train_features': train_features, train_labels': train_labels, 'test_features': test_features, 'test_labels': test_labels,}
            criterion (torch.nn.Loss): Loss class from PyTorch NN module.
                    Example: torch.nn.MSELoss()
            epochs (int): Number of training epochs
            batch_size (int): Number of training batches
            tensorboard (bool, optional): Flag determining whether Tensorboard logs should be generated and outputted. Defaults to False.
            num_linear_layers (int, optional): Number of linear layers in the model. Only used if paired with ExploratoryModel. Defaults to None.
            nodes (list, optional): List of integers denoting the number of nodes in num_linear_layers. Len(nodes) must equal num_linear_layers. Defaults to None.
            save_model (bool, optional): Flag determining whether the trained model should be saved. Defaults to False.
            performance_optimized (bool, optional): Flag determining whether the training loop should be optimized for fast training. Defaults to False.
        """
        
        # save attributes
        self.data_dict = data_dict
        self.num_input_features = self.data_dict['train_features'].shape[1]
        
        # Loop through possible architecture parameters and if it not given, set it to None
        for param in ['num_linear_layers', 'nodes', 'num_rnn_hidden', 'num_rnn_layers']:
            try:
                architecture[param]
            except:
                architecture[param] = None
        architecture['input_layer_size'] = self.num_input_features

        # establish model - if using exploratory model, use num_linear_layers and nodes arg
        self.model = model(architecture=architecture).to(self.device)
        self.time_series = True if hasattr(self.model, 'time_series') else False

        # If the data loader hasn't been created, run _format_data function
        if self.train_loader is None or self.train_loader is None:
            self._format_data(data_dict['train_features'], data_dict['train_labels'], data_dict['test_features'],
                              data_dict['test_labels'],
                              train_batch_size=batch_size)

        # Use multiple GPU parallelization if available
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

        optimizer = optim.Adam(self.model.parameters(), )
        # criterion = nn.MSELoss()
        self.time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")

        comment = f" -- {self.time}, FC={architecture['num_linear_layers']}, nodes={architecture['nodes']}, batch_size={batch_size},"
        # comment = f" -- {self.time}, dataset={dataset},"
        tb = SummaryWriter(comment=comment)
        mae = nn.L1Loss()
        X_test = torch.tensor(self.X_test, dtype=torch.float).to(self.device)
        y_test = torch.tensor(self.y_test, dtype=torch.float).to(self.device)
        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            total_loss = 0
            total_mae = 0
            for X_train_batch, y_train_batch in self.train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)

                optimizer.zero_grad()

                pred = self.model(X_train_batch)
                loss = criterion(pred, y_train_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                self.logs['training']['batch'].append(loss.item())

                if not performance_optimized:
                    total_mae += mae(pred, y_train_batch.unsqueeze(1)).item()

            avg_mse = total_loss / len(self.train_loader)
            self.logs['training']['epoch'].append(avg_mse)

            if not performance_optimized:
                avg_rmse = np.sqrt(avg_mse)
                avg_mae = total_mae / len(self.train_loader)

            training_end = time.time()

            if not performance_optimized:
                self.model.eval()
                test_total_loss = 0
                test_total_mae = 0
                for X_test_batch, y_test_batch in self.test_loader:
                    X_test_batch, y_test_batch = X_test_batch.to(self.device), y_test_batch.to(self.device)
                    test_pred = self.model(X_test_batch)
                    loss = criterion(test_pred, y_test_batch.unsqueeze(1))
                    test_total_loss += loss.item()
                    test_total_mae += mae(test_pred, y_test_batch.unsqueeze(1)).item()

                test_mse = test_total_loss / len(self.test_loader)
                test_mae = test_total_mae / len(self.test_loader)
                self.logs['testing'].append(test_mse)
                testing_end = time.time()

                preds = self.model(X_test).to(device)
                if self.device.type != 'cuda':
                    r2 = r2_score(self.y_test, preds.detach().numpy())
                else:
                    r2 = r2_score(self.y_test, preds.cpu().detach().numpy())

            if tensorboard:
                tb.add_scalar("Training MSE", avg_mse, epoch)

                if not performance_optimized:
                    tb.add_scalar("Training RMSE", avg_rmse, epoch)
                    tb.add_scalar("Training MAE", avg_mae, epoch)

                    tb.add_scalar("Validation MSE", test_mse, epoch)
                    tb.add_scalar("Validation RMSE", np.sqrt(test_mse), epoch)
                    tb.add_scalar("Validation MAE", test_mae, epoch)
                    tb.add_scalar("R^2", r2, epoch)

            # TODO: Add verbose parameter so you can turn these off
            if not performance_optimized:
                print('')
                print(f"""Epoch: {epoch}/{epochs}, Training Loss (MSE): {avg_mse:0.8f}, Validation Loss (MSE): {test_mse:0.8f}
Training time: {training_end - epoch_start: 0.2f} seconds, Validation time: {testing_end - training_end: 0.2f} seconds""")

            else:
                print('')
                print(f"""Epoch: {epoch}/{epochs}, Training Loss (MSE): {avg_mse:0.8f}
Training time: {training_end - epoch_start: 0.2f} seconds""")

        if tensorboard:
            metrics, _ = self.evaluate()
            tb.add_hparams(
                {"FC": architecture['num_linear_layers'], "nodes": architecture['nodes'], "batch_size": batch_size, },

                {
                    "Test MSE": metrics['MSE'], "Test MAE": metrics['MAE'], "R^2": metrics['R2'],
                    "RMSE": metrics["RMSE"]
                },
            )

            tb.close()

        # TODO: Change to relative path
        if save_model:
            torch.save(self.model.state_dict(), f"/users/pvankatw/emulator/src/models/experiment_models/{self.time}.pt")

        return self

    def plot_loss(self, save=False):
        plt.plot(self.logs['training']['epoch'], 'r-', label='Training Loss')
        plt.plot(self.logs['testing'], 'b-', label='Validation Loss')
        plt.title('Emulator MSE loss per Epoch')
        plt.xlabel(f'Epoch #')
        plt.ylabel('Loss (MSE)')
        plt.legend()

        if save:
            plt.savefig(save)
        plt.show()

    def evaluate(self):
        # Test predictions
        self.model.eval()
        preds = torch.tensor([]).to(self.device)
        for X_test_batch, y_test_batch in self.test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(self.device), y_test_batch.to(self.device)
            test_pred = self.model(X_test_batch)
            preds = torch.cat((preds, test_pred), 0)

        if self.device.type == 'cuda':
            preds = preds.squeeze().cpu().detach().numpy()
        else:
            preds = preds.squeeze().detach().numpy()
            
        mse = sum((preds - self.y_test)**2) / len(preds)
        mae = sum((preds - self.y_test)) / len(preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, preds)

        metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

        print(f"""Test Metrics
MSE: {mse:0.6f}
MAE: {mae:0.6f}
RMSE: {rmse:0.6f}
R2: {r2:0.6f}""")

        return metrics, preds
