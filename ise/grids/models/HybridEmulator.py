import torch
from torch import nn, optim
from nflows import distributions, flows, transforms
import numpy as np
from sklearn.decomposition import PCA
import pickle as pkl
from ise.grids.data.EmulatorDataset import EmulatorDataset
import warnings
import pandas as pd
import os
from ise.grids.data.process import DimensionalityReducer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def total_variation_regularization(grid, ):
    # Calculate the sum of horizontal and vertical differences
    horizontal_diff = np.abs(np.diff(grid, axis=1))
    vertical_diff = np.abs(np.diff(grid, axis=0))
    total_variation = np.sum(horizontal_diff) + np.sum(vertical_diff)
    return total_variation

def spatial_loss(true, predicted, smoothness_weight=0.2):
    pixelwise_mse = sum(sum(abs(true-predicted)))/true.size
    tvr = total_variation_regularization(predicted,)
    return pixelwise_mse + smoothness_weight*tvr


def combined_loss(true, predicted, x, y, flow, predictor_weight=0.5, nf_weight=0.5,):
    if predictor_weight + nf_weight != 1:
        raise ValueError("The sum of predictor_weight and nf_weight must be 1")
    predictor_loss = spatial_loss(true, predicted, smoothness_weight=0.2)
    nf_loss = -flow.log_prob(inputs=y, context=x)
    return predictor_weight*predictor_loss + nf_weight*nf_loss
        
class ScalerModel:
    def __init__(self, scaler_model):
        if isinstance(scaler_model, str):
            self.model = self.load_model(scaler_model)
        elif isinstance(scaler_model, StandardScaler):
            self.model = scaler_model
        else:
            raise ValueError("scaler_model must be a path (str) or a StandardScaler instance")
        
    def load_model(self, path, ):
        return pkl.load(open(path, "rb"))
    
    def invert(self, data):
        return self.model.inverse_transform(data)

class PCAModel:
    def __init__(self, pca_model, scaler_path,):
        if isinstance(pca_model, str):
            self.model = self.load_model(pca_model)
        elif isinstance(pca_model, PCA):
            self.model = pca_model
        else:
            raise ValueError("pca_model must be a path (str) or a PCA instance")
        self.scaler = ScalerModel(scaler_path)
        
    def load_model(self, path, ):
        return pkl.load(open(path, "rb"))
        
    def invert(self, pcs):
        inverted = self.model.inverse_transform(pcs)
        unscaled = self.scaler.invert(inverted)
        return unscaled
    
        

class WeakPredictor(nn.Module):
    def __init__(self, input_size, lstm_num_layers, lstm_hidden_size, output_size, pca_model, scaler_path=None):
        super(WeakPredictor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = lstm_hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            num_layers=lstm_num_layers,
        )
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=lstm_hidden_size, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.linear_out = nn.Linear(in_features=256, out_features=output_size)
        
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = spatial_loss
        self.trained = False
        
        if isinstance(pca_model, PCAModel):
            self.pca_model = pca_model
        elif isinstance(pca_model, str) and scaler_path is None:
            raise ValueError("If pca_model is a path, scaler_path must be provided")
        elif isinstance(pca_model, str) and scaler_path is not None:
            self.pca_model = PCAModel(pca_model, scaler_path)
        else:
            raise ValueError("pca_model must be a PCAModel instance or a path (str)")
        

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
        x = self.linear2(x)
        x = self.relu(x)
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
        
        for epoch in range(1, epochs+1):
            print(f"Epoch: {epoch}")
            for i, (x, y) in enumerate(data_loader):
                self.optimizer.zero_grad()
                y_pred = self.forward(x)
                y_pred = self.pca_model.invert(y_pred)
                y = self.pca_model.invert(y)
                loss = self.criterion(y, y_pred)
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        self.trained = True
    
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
    def __init__(self,):
        super(NormalizingFlow, self).__init__()
        self.num_flow_transforms = 10
        self.flow_hidden_features = 100
        self.num_input_features = 10
        self.num_predicted_sle = 10
        
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
        self.trained = True
        
    def get_latent(self, x, latent_constant=0.0):
        return self.flow.t(0, context=x)
    
    def aleatoric(self, inputs, context, num_samples):
        samples = self.flow.sample(num_samples, context)
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
        mean, epistemic = self.deep_ensemble(x)
        aleatoric = self.normalizing_flow.aleatoric(x, mean, 100)
        return mean, epistemic, aleatoric