"""Testing functions for analyzing performance of pretrained models."""
import torch
import pandas as pd
from ise.models.training.Trainer import Trainer
import numpy as np
np.random.seed(10)
from sklearn.metrics import r2_score
from ise.utils.data import load_ml_data

def test_pretrained_model(model_path: str, model_class, architecture: dict, data_directory: str, 
                          time_series: bool, mc_dropout: bool=False, dropout_prob: float=0.1, 
                          mc_iterations: int=100, verbose: bool=True):
    """Runs testing procedure on a pretrained and saved model. Makes model predictions and tests
    them based on standard metrics. Outputs the metrics in a dictionary as well as the predictions.

    Args:
        model_path (str): Path to the pretrained model. Must be a '.pt' model.
        model_class (ModelClass): Model class used to train the model.
        architecture (dict): Architecture arguments used to train the model. 
        data_directory (str): Directory containing training and testing data.
        time_series (bool): Flag denoting wether model was trained with time-series data.
        mc_dropout (bool, optional): Flag denoting whether the model was trained with MC dropout protocol. Defaults to False.
        dropout_prob (float, optional): Dropout probability in MC dropout protocol. Unused if mc_dropout=False. Defaults to 0.1.
        mc_iterations (int, optional): MC iterations to be used in testing. Unused if mc_dropout=False. Defaults to 100.
        verbose (bool, optional): Flag denoting whether to output logs to terminal. Defaults to True.
    
    Returns:
        tuple: Tuple containing [metrics, preds, bounds], or test metrics, predictions, and uncertainty bounds on test_features.
    """    
    
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
    
    upper_q = quantiles[1,:]
    lower_q = quantiles[0,:]
    bounds = {'upper_ci': upper_ci, 'lower_ci': lower_ci,
              'upper_q': upper_q, 'lower_q': lower_q}
        
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

    return metrics, preds, bounds

def mc_accuracy(model_path: str, model_class, architecture: dict, data_directory: str, 
                time_series: bool, dropout_prob: float=0.1, mc_iterations: int=30, verbose: bool=True):
    """Tests the accuracy of the MC dropout uncertainty bands. Shows the proportion of true 
    values that fall within the uncertainty range.

    Args:
        model_path (str): Path to the pretrained model. Must be a '.pt' model.
        model_class (ModelClass): Model class used to train the model.
        architecture (dict): Architecture arguments used to train the model. 
        data_directory (str): Directory containing training and testing data.
        time_series (bool): Flag denoting wether model was trained with time-series data.
        dropout_prob (float, optional): Dropout probability in MC dropout protocol. Unused if mc_dropout=False. Defaults to 0.1.
        mc_iterations (int, optional): MC iterations to be used in testing. Unused if mc_dropout=False. Defaults to 30.
        verbose (bool, optional): Flag denoting whether to output logs to terminal. Defaults to True.

    Returns:
        tuple: Tuple containing [ci_accuracy, q_accuracy], or confidence interval accuracy and quantile accuracy
    """
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
    
    # Predict on test set and return confidence intervals and quantiles
    all_preds, means, upper_ci, lower_ci, quantiles = model.predict(X_test, mc_iterations=mc_iterations)
    upper_q = quantiles[1,:]
    lower_q = quantiles[0,:]
    preds = means
    
    # Get accuracy based on how many true values fall between CI and Q.
    test_labels = np.array(test_labels).squeeze()
    q_acc = ((test_labels >= lower_q) & (test_labels <= upper_q)).mean()
    ci_acc = ((test_labels >= lower_ci) & (test_labels <= upper_ci)).mean()
    
    return ci_acc, q_acc

        
    
    