from ise.models.training.Trainer import Trainer
from ise.models.traditional import ExploratoryModel
from ise.models.timeseries.TimeSeriesEmulator import TimeSeriesEmulator
from ise.models.traditional.ExploratoryModel import ExploratoryModel
from torch import nn
import pandas as pd
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
np.random.seed(10)
from sklearn.metrics import r2_score


def train_timeseries_network(data_directory, 
                             architecture=None, 
                             epochs=20, 
                             batch_size=100, 
                             model=TimeSeriesEmulator,
                             loss=nn.MSELoss(),
                             tensorboard=False,
                             save_model=False,
                             performance_optimized=False,
                             verbose=False,
                             tensorboard_comment=None
                             ):
    
    if verbose:
        print('1/3: Loading processed data...')
    try:
        test_features = pd.read_csv(f'{data_directory}/ts_test_features.csv')
        train_features = pd.read_csv(f'{data_directory}/ts_train_features.csv')
        test_labels = pd.read_csv(f'{data_directory}/ts_test_labels.csv')
        train_labels = pd.read_csv(f'{data_directory}/ts_train_labels.csv')
        scenarios = pd.read_csv(f'{data_directory}/ts_test_scenarios.csv').values.tolist()
    except FileNotFoundError:
            raise FileNotFoundError('Files not found. Format must be in format \"ts_train_features.csv\"')
        
    data_dict = {'train_features': train_features,
                'train_labels': train_labels,
                'test_features': test_features,
                'test_labels': test_labels, }
    
    trainer = Trainer()
    if verbose:
        print('2/3: Training Model...')
        
    if architecture is None:
        architecture = {
            'num_rnn_layers': 4,
            'num_rnn_hidden': 128,
        }
    
    print('Architecture: ')
    print(architecture)

    trainer.train(
        model=model,
        architecture=architecture,
        data_dict=data_dict,
        criterion=loss,
        epochs=epochs,
        batch_size=batch_size,
        tensorboard=tensorboard,
        save_model=save_model,
        performance_optimized=performance_optimized,
        sequence_length=5,
        verbose=verbose,
        tensorboard_comment=tensorboard_comment,
    )

    if verbose:
        print('3/3: Evaluating Model')
    model = trainer.model
    metrics, test_preds = trainer.evaluate(verbose=verbose)
    return model, metrics, test_preds


def train_traditional_network(data_directory, 
                             architecture=None, 
                             epochs=20, 
                             batch_size=100, 
                             model=ExploratoryModel,
                             loss=nn.MSELoss(),
                             tensorboard=False,
                             save_model=False,
                             performance_optimized=False,
                             verbose=False,
                             tensorboard_comment=None
                             ):
    
    if verbose:
        print('1/3: Loading processed data...')
    
    try:
        test_features = pd.read_csv(f'{data_directory}/traditional_test_features.csv')
        train_features = pd.read_csv(f'{data_directory}/traditional_train_features.csv')
        test_labels = pd.read_csv(f'{data_directory}/traditional_test_labels.csv')
        train_labels = pd.read_csv(f'{data_directory}/traditional_train_labels.csv')
        scenarios = pd.read_csv(f'{data_directory}/traditional_test_scenarios.csv').values.tolist()
    except FileNotFoundError:
            raise FileNotFoundError('Files not found. Format must be in format \"traditional_train_features.csv\"')
    
    if 'lag' in train_features.columns:
        raise AttributeError('Data must be processed using timeseries=True in feataure_engineering. Rerun feature engineering to train traditional network.')
        
    data_dict = {'train_features': train_features,
                'train_labels': train_labels,
                'test_features': test_features,
                'test_labels': test_labels, }
    
    trainer = Trainer()
    if verbose:
        print('2/3: Training Model...')
        
    if architecture is None:
        architecture = {
            'num_linear_layers': 4,
            'nodes': [128, 64, 32, 1],
        }

    trainer.train(
        model=model,
        architecture=architecture,
        data_dict=data_dict,
        criterion=loss,
        epochs=epochs,
        batch_size=batch_size,
        tensorboard=tensorboard,
        save_model=save_model,
        performance_optimized=performance_optimized,
        sequence_length=5,
        verbose=verbose,
        tensorboard_comment=tensorboard_comment,
    )

    if verbose:
        print('3/3: Evaluating Model')
    model = trainer.model
    metrics, test_preds = trainer.evaluate(verbose=verbose)
    return metrics, test_preds

def train_gaussian_process(data_directory, n, features=['temperature'], sampling_method='random', kernel=None, verbose=False):
    
    if verbose:
        print('1/3: Loading processed data...')
    
    try:
        test_features = pd.read_csv(f'{data_directory}/traditional_test_features.csv')
        train_features = pd.read_csv(f'{data_directory}/traditional_train_features.csv')
        test_labels = pd.read_csv(f'{data_directory}/traditional_test_labels.csv')
        train_labels = pd.read_csv(f'{data_directory}/traditional_train_labels.csv')
        scenarios = pd.read_csv(f'{data_directory}/traditional_test_scenarios.csv').values.tolist()
    except FileNotFoundError:
        test_features = pd.read_csv(f'{data_directory}/ts_test_features.csv')
        train_features = pd.read_csv(f'{data_directory}/ts_train_features.csv')
        test_labels = pd.read_csv(f'{data_directory}/ts_test_labels.csv')
        train_labels = pd.read_csv(f'{data_directory}/ts_train_labels.csv')
        scenarios = pd.read_csv(f'{data_directory}/ts_test_scenarios.csv').values.tolist()
        
    if kernel is None:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    
    gaussian_process = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=9
    )
    
    if not isinstance(features, list):
        raise ValueError(f'features argument must be a list, received {type(features)}')
    
    if sampling_method.lower() == 'random':
        gp_train_features = train_features[features].sample(n)
    elif sampling_method.lower() == 'first_n':
        gp_train_features = train_features[features][:n]
    else:
        raise ValueError(f'sampling method must be in [random, first_n], received {sampling_method}')
    
    gp_train_labels = np.array(train_labels.loc[gp_train_features.index]).reshape(-1, 1)
    gp_test_features = test_features[features]
    if isinstance(gp_test_features, pd.Series) or gp_test_features.shape[1] == 1:
        gp_test_features = np.array(gp_test_features).reshape(-1, 1)
    
    gaussian_process.fit(gp_train_features, gp_train_labels,)
    preds, std_prediction = gaussian_process.predict(gp_test_features, return_std=True)
    
    mse = sum((preds - test_labels)**2) / len(preds)
    mae = sum(abs((preds - test_labels))) / len(preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_labels, preds)
    
    metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

    if verbose:
        print(f"""Test Metrics
MSE: {mse:0.6f}
MAE: {mae:0.6f}
RMSE: {rmse:0.6f}
R2: {r2:0.6f}""")
    
    return preds, std_prediction, metrics