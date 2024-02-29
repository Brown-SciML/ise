import torch
from torch import nn, optim
from nflows import distributions, flows, transforms
import numpy as np
from ise.grids.data.EmulatorDataset import EmulatorDataset
import warnings
import pandas as pd
from ise.grids.models.PCA import PCA
from ise.grids.models.Scaler import Scaler 

class Criterion(nn.Module):
    def __init__(self,):
        super(Criterion, self).__init__()
        
    def total_variation_regularization(self, grid, ):
        # Calculate the sum of horizontal and vertical differences
        horizontal_diff = torch.abs(torch.diff(grid, axis=2))
        vertical_diff = torch.abs(torch.diff(grid, axis=1))
        total_variation = torch.sum(horizontal_diff, axis=(1,2)) + torch.sum(vertical_diff, axis=(1,2))
        return torch.mean(total_variation)

    def spatial_loss(self, true, predicted, smoothness_weight=0.001):
        pixelwise_mse = torch.mean(torch.abs(true - predicted)**2,) # loss for each image in the batch (batch_size,)
        tvr = self.total_variation_regularization(predicted,)
        return pixelwise_mse + smoothness_weight * tvr


    def forward(self, true, predicted, x, y, flow, predictor_weight=0.5, nf_weight=0.5,):
        if predictor_weight + nf_weight != 1:
            raise ValueError("The sum of predictor_weight and nf_weight must be 1")
        predictor_loss = self.spatial_loss(true, predicted, smoothness_weight=0.2)
        nf_loss = -flow.log_prob(inputs=y, context=x)
        return predictor_weight*predictor_loss + nf_weight*nf_loss
        

class DimensionProcessor(nn.Module):
    def __init__(self, pca_model, scaler_model,):
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
        if isinstance(scaler_model, str):
            self.scaler = Scaler.load(scaler_model)
        elif isinstance(scaler_model, PCA):
            self.scaler = scaler_model
        else:
            raise ValueError("pca_model must be a path (str) or a PCA instance")
        if self.scaler.mean_ is None or self.scaler.scale_ is None:
            raise RuntimeError("This StandardScalerPyTorch instance is not fitted yet.")

        self.scaler.to(self.device)
        self.pca.to(self.device)
        self.to(self.device)
    
    def to_pca(self, data):
        data = data.to(self.device)
        scaled = self.scaler.transform(data) # scale
        return self.pca.transform(scaled) # convert to pca
        
    def to_grid(self, pcs):
        if not isinstance(pcs, torch.Tensor):
            pcs = torch.tensor(pcs, dtype=torch.float32).to(self.device)
        else:
            pcs = pcs.to(self.device)
        # Ensure components and mean are on the same device as pcs
        components = self.pca.components.to(self.device)
        pca_mean = self.pca.mean.to(self.device)
        # Now, the operation should not cause a device mismatch error
        scaled_grid = torch.mm(pcs, components.t()) + pca_mean
        
        scale = self.scaler.scale_.to(self.device)
        scaler_mean = self.scaler.mean_.to(self.device)
        unscaled_grid = scaled_grid * scale + scaler_mean
        return unscaled_grid
        

class WeakPredictor(nn.Module):
    def __init__(self, input_size, lstm_num_layers, lstm_hidden_size, output_size, dim_processor, scaler_path=None, ice_sheet='AIS'):
        super(WeakPredictor, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = lstm_hidden_size
        self.ice_sheet = ice_sheet
        self.ice_sheet_dim = (761, 761) if ice_sheet == 'AIS' else (337, 577)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
            
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            num_layers=lstm_num_layers,
        )
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=lstm_hidden_size, out_features=128)
        # self.linear2 = nn.Linear(in_features=128, out_features=64)
        self.linear_out = nn.Linear(in_features=128, out_features=output_size)
        
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = Criterion().to(self.device)
        self.trained = False
        
        if isinstance(dim_processor, DimensionProcessor):
            self.dim_processor = dim_processor
        elif isinstance(dim_processor, str) and scaler_path is None:
            raise ValueError("If dim_processor is a path to a PCA object, scaler_path must be provided")
        elif isinstance(dim_processor, str) and scaler_path is not None:
            self.dim_processor = DimensionProcessor(pca_model=self.pca_model, scaler_model=scaler_path)
        else:
            raise ValueError("dim_processor must be a DimensionProcessor instance or a path (str) to a PCA object with scaler_path specified as a Scaler object.")
        
        self.dim_processor.to(self.device)

    def forward(self, x):
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
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        x = hn[-1, :, :] # last layer of the hidden state
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu(x)
        x = self.linear_out(x)
        
        return x
    
    def fit(self, X, y, epochs=100, sequence_length=3, batch_size=64):
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
            
        dataset = EmulatorDataset(X, y, sequence_length=sequence_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        self.to(self.device)
        
        for epoch in range(1, epochs+1):
            print(f"Epoch: {epoch}")
            for i, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.shape[0]
                self.optimizer.zero_grad()
                y_pred = self.forward(x)
                y_pred = self.dim_processor.to_grid(y_pred).reshape(batch_size, self.ice_sheet_dim[0], self.ice_sheet_dim[1])
                y_true = self.dim_processor.to_grid(y).reshape(batch_size, self.ice_sheet_dim[0], self.ice_sheet_dim[1])
                
                # yp = y_pred[0, :, :], yt = y_true[0, :, :]
                loss = self.criterion.spatial_loss(y_true, y_pred, smoothness_weight=0.0)
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        self.trained = True
        
    def predict(self, X, sequence_length=3, batch_size=64):
        self.eval()
        self.to(self.device)
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        dataset = EmulatorDataset(X, y=None, sequence_length=sequence_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        X = torch.tensor(X).to(self.device)
        
        preds = torch.tensor([]).to(self.device)
        for X_test_batch in data_loader:
            self.eval()
            X_test_batch = X_test_batch.to(self.device)
            y_pred = self.forward(X_test_batch)
            y_pred = self.dim_processor.to_grid(y_pred)
            preds = torch.cat((preds, y_pred), 0)
        
        return preds
    
class DeepEmulator(nn.Module):
    def __init__(self, weak_predictors: list=[]):
        super(DeepEmulator, self).__init__()
        
        if not weak_predictors:
            self.weak_predictors = [
                WeakPredictor(
                    lstm_num_layers=np.random.choice(4, 1), 
                    lstm_hidden_size=np.random.choice([512, 256, 128, 64], 1)
                    ) 
                for _ in range(10)
                ]
        else:
            if isinstance(weak_predictors, list):
                self.weak_predictors = weak_predictors
            else:
                raise ValueError("weak_predictors must be a list of WeakPredictor instances")
            
            if any([x for x in weak_predictors if not isinstance(x, WeakPredictor)]):
                raise ValueError("weak_predictors must be a list of WeakPredictor instances")
        
        # check to see if all weak predictors are trained
        self.trained = all([wp.trained for wp in self.weak_predictors])
        

    def forward(self, x):
        if not self.trained:
            warnings.warn("This model has not been trained. Predictions will not be accurate.")
        mean_prediction = np.mean([wp(x) for wp in self.weak_predictors], axis=0)
        epistemic_uncertainty = np.var([wp(x) for wp in self.weak_predictors], axis=0)
        return mean_prediction, epistemic_uncertainty
    
    def fit(self, X, y, epochs=100, batch_size=64):
        if self.trained:
            warnings.warn("This model has already been trained. Training anyways.")
        for wp in self.weak_predictors:
            wp.fit(X, y, epochs, batch_size)
        
    
    
    
class NormalizingFlow(nn.Module):
    def __init__(self, forcing_size, sle_size,):
        super(NormalizingFlow, self).__init__()
        self.num_flow_transforms = 5
        self.flow_hidden_features = 128
        self.num_input_features = forcing_size
        self.num_predicted_sle = sle_size
        
        self.base_distribution = distributions.normal.ConditionalDiagonalNormal(
            shape=[self.num_input_features],
            context_encoder=nn.Linear(1, self.num_input_features * 2)
        )
        
        t = []
        for _ in range(self.num_flow_transforms):
            t.append(transforms.permutations.RandomPermutation(features=self.num_predicted_sle, ))
            t.append(transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=self.num_predicted_sle,
                hidden_features=self.flow_hidden_features,
                context_features=self.num_input_features,
            ))

        self.t = transforms.base.CompositeTransform(t)

        self.flow = flows.base.Flow(
            transform=self.t,
            distribution=self.base_dist
        )

        self.optimizer = optim.Adam(self.flow.parameters())
        self.criterion = -self.flow.log_prob
        self.trained = False
    
    def fit(self, X, y, epochs=100, batch_size=64):
        dataset = EmulatorDataset(X, y, sequence_length=3)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        
        for epoch in range(1, epochs+1):
            for i, (x, y) in enumerate(data_loader):
                self.optimizer.zero_grad()
                loss = self.criterion(inputs=y, context=x)
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        self.trained = True
        
    def get_latent(self, x, latent_constant=0.0):
        return self.flow.t(latent_constant, context=x)
    
    def aleatoric(self, inputs, num_samples):
        samples = self.flow.sample(num_samples, inputs)
        samples = samples.detach().numpy()
        variance = np.var(samples, axis=0)
        return variance
        

class HybridEmulator(torch.nn.Module):
    def __init__(self, deep_ensemble, normalizing_flow):
        super(HybridEmulator, self).__init__()
        
        if not isinstance(deep_ensemble, DeepEmulator):
            raise ValueError("deep_ensemble must be a DeepEmulator instance")
        if not isinstance(normalizing_flow, NormalizingFlow):
            raise ValueError("normalizing_flow must be a NormalizingFlow instance")
        
        self.deep_ensemble = deep_ensemble
        self.normalizing_flow = normalizing_flow
        self.trained = self.deep_ensemble.trained and self.normalizing_flow.trained
    
    def train(self, X, y, ):
        if self.trained:
            warnings.warn("This model has already been trained. Training anyways.") 
        self.normalizing_flow.fit(X, y, epochs=250)
        z = self.normalizing_flow.get_latent(X, y)
        self.deep_ensemble.fit(z, y)
        self.trained = True
        
    def forward(self, x):
        if not self.trained:
            warnings.warn("This model has not been trained. Predictions will not be accurate.")
        prediction, epistemic = self.deep_ensemble(x)
        aleatoric = self.normalizing_flow.aleatoric(x, 100)
        return prediction, epistemic, aleatoric