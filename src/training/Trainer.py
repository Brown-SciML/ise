from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from training.PyTorchDataset import PyTorchDataset
from torch.utils.data import DataLoader
import time



class Trainer:
    def __init__(self, cfg):
        self.model = None
        self.data = {}
        self.cfg = cfg
        # self.data_dir = self.cfg['data']['directory'] if self.cfg['data']['directory'] is not None else self.cfg['data']['export']
        # self.model_name = cfg['training']['model']
        # self.num_epochs = cfg['training']['epochs']
        # self.verbose = cfg['training']['verbose']
        # self.batch_size = cfg['training']['batch_size']
        self.num_input_features = None
        self.loss_logs = {'training_loss': [], 'val_loss': [],}
        self.train_loader = None
        self.test_loader = None
        self.data_dict = None
        self.logs = {'training': {'epoch': [], 'batch': []}, 'testing': []}

    def _format_data(self, train_features, train_labels, test_features, test_labels, train_batch_size=100, test_batch_size=10,):
        self.X_train = np.array(train_features, dtype=np.float64)
        self.y_train = np.array(train_labels, dtype=np.float64)
        self.X_test = np.array(test_features, dtype=np.float64)
        self.y_test = np.array(test_labels, dtype=np.float64)

        train_dataset = PyTorchDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).float().squeeze())
        test_dataset = PyTorchDataset(torch.from_numpy(self.X_test).float(), torch.from_numpy(self.y_test).float().squeeze())
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,)
        
        return self

    def train(self, model, data_dict, criterion, epochs, batch_size, tensorboard=False, num_linear_layers=None, nodes=None,):
        self.data_dict = data_dict
        if self.train_loader is None or self.train_loader is None:
            self._format_data(data_dict['train_features'], data_dict['train_labels'], data_dict['test_features'], data_dict['test_labels'],
                              train_batch_size=batch_size)
        
        self.num_input_features = self.data_dict['train_features'].shape[1]
        
        
        if num_linear_layers is not None and nodes is not None:
            self.model = model(input_layer_size=self.num_input_features, num_linear_layers=num_linear_layers, nodes=nodes)
        else:
            self.model = model(input_layer_size=self.num_input_features)
        
        optimizer = optim.Adam(self.model.parameters(),)
        # criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(1, epochs+1):
            epoch_start = time.time()

            total_loss = 0
            for X_train_batch, y_train_batch in self.train_loader:
                
                optimizer.zero_grad()

                pred = self.model(X_train_batch)
                loss = criterion(pred, y_train_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                self.logs['training']['batch'].append(loss.item())
                
            avg_loss = total_loss / len(self.train_loader)
            self.logs['training']['epoch'].append(avg_loss)
            training_end = time.time()
            
            self.model.eval()
            test_total_loss = 0
            for X_test_batch, y_test_batch in self.test_loader:
                test_pred = self.model(X_test_batch)
                loss = criterion(test_pred, y_test_batch.unsqueeze(1))
                test_total_loss += loss.item()
                
            test_loss = test_total_loss / len(self.test_loader)
            self.logs['testing'].append(test_loss)
            testing_end = time.time()

            if epoch % 1 == 0:
                print('')
                print(f"""Epoch: {epoch}/{epochs}, Training Loss (MSE): {avg_loss:0.8f}, Validation Loss (MSE): {test_loss:0.8f}
Training time: {training_end - epoch_start: 0.2f} seconds, Validation time: {testing_end - training_end: 0.2f} seconds""")


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

        X_test = torch.tensor(self.X_test, dtype=torch.float)  # .reshape(-1, X_train.size()[2])
        preds = self.model(X_test)

        # Calculate metrics
        mae = mean_absolute_error(self.y_test, preds.detach().numpy())
        mse = mean_squared_error(self.y_test, preds.detach().numpy())
        rmse = np.sqrt(mean_squared_error(self.y_test, preds.detach().numpy()))
        r2 = r2_score(self.y_test, preds.detach().numpy())
        
        metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

        print(f"""Test Metrics
MSE: {mse:0.6f}
MAE: {mae:0.6f}
RMSE: {rmse:0.6f}
R2: {r2:0.6f}""")
        
        return metrics, preds