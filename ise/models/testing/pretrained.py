import torch
import pandas as pd
from ise.models.training.Trainer import Trainer
import numpy as np
np.random.seed(10)
from sklearn.metrics import r2_score
from ise.utils.utils import load_ml_data

def test_pretrained_model(model_path, model_class, architecture, data_directory, time_series, mc_dropout=False, dropout_prob=0.1, mc_iterations=100, verbose=True):
    
    if verbose:
        print('1/3: Loading processed data...')
    
    train_features, train_labels, test_features, test_labels, test_scenarios = load_ml_data(data_directory=data_directory, time_series=time_series)
            
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data_dict = {'train_features': train_features,
                'train_labels': train_labels,
                'test_features': test_features,
                'test_labels': test_labels, }
    
    # Load Model
    trainer = Trainer()
    sequence_length = 5 if time_series else None
    trainer._initiate_model(model_class, data_dict=data_dict, architecture=architecture, sequence_length=sequence_length, batch_size=100, mc_dropout=mc_dropout, dropout_prob=dropout_prob)
    
    if verbose:
        print('2/3: Loading pretrained weights...')
    # Assigned pre-trained weights
    trainer.model.load_state_dict(torch.load(model_path, map_location=device))
    model = trainer.model
    
    # Evaluate on test_features
    if verbose:
        print('3/3: Evaluating...')
    model.eval()
    X_test = torch.from_numpy(np.array(test_features, dtype=np.float64)).float()
    
    if mc_dropout:
        all_preds, means, upper_ci, lower_ci, quantiles = model.predict(X_test, mc_iterations=mc_iterations)
        preds = means
    else:
        preds, means, upper_ci, lower_ci, quantiles = model.predict(X_test, mc_iterations=1)
        
    test_labels = np.array(test_labels).squeeze()
    mse = sum((preds - test_labels)**2) / len(preds)
    mae = sum(abs((preds - test_labels))) / len(preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(np.array(test_labels), preds)
    
    metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print(f"""Test Metrics
MSE: {mse:0.6f}
MAE: {mae:0.6f}
RMSE: {rmse:0.6f}
R2: {r2:0.6f}""")

    return metrics, preds

def mc_accuracy(model_path, model_class, architecture, data_directory, time_series, dropout_prob=0.1, mc_iterations=30, verbose=True):
    if verbose:
        print('1/3: Loading processed data...')
    
    train_features, train_labels, test_features, test_labels, test_scenarios = load_ml_data(data_directory=data_directory, time_series=time_series)
            
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data_dict = {'train_features': train_features,
                'train_labels': train_labels,
                'test_features': test_features,
                'test_labels': test_labels, }
    
    # Load Model
    trainer = Trainer()
    sequence_length = 5 if time_series else None
    trainer._initiate_model(model_class, data_dict=data_dict, architecture=architecture, sequence_length=sequence_length, batch_size=100, mc_dropout=True, dropout_prob=dropout_prob)
    
    if verbose:
        print('2/3: Loading pretrained weights...')
    # Assigned pre-trained weights
    trainer.model.load_state_dict(torch.load(model_path, map_location=device))
    model = trainer.model
    
    # Evaluate on test_features
    if verbose:
        print('3/3: Evaluating...')
    model.eval()
    X_test = torch.from_numpy(np.array(test_features, dtype=np.float64)).float()
    
    all_preds, means, upper_ci, lower_ci, quantiles = model.predict(X_test, mc_iterations=mc_iterations)
    upper_q = quantiles[1,:]
    lower_q = quantiles[0,:]
    preds = means
    
    test_labels = np.array(test_labels).squeeze()
    q_acc = ((test_labels >= lower_q) & (test_labels <= upper_q)).mean()
    ci_acc = ((test_labels >= lower_ci) & (test_labels <= upper_ci)).mean()
    
    return ci_acc, q_acc

        
    
    