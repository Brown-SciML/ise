import yaml
import os
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(10)


file_dir = os.path.dirname(os.path.realpath(__file__))

def get_configs():
    # Loads configuration file in the repo and formats it as a dictionary.
    try:
        with open(f'{file_dir}/config.yaml') as c:
            data = yaml.load(c, Loader=yaml.FullLoader)
    except FileNotFoundError:  
        try:   # depends on where you're calling it from...
            with open('./config.yaml') as c:
                data = yaml.load(c, Loader=yaml.FullLoader)
        except FileNotFoundError:
            with open('config.yaml') as c:
                data = yaml.load(c, Loader=yaml.FullLoader)
    return data


def check_input(input, options, argname=None):
    # simple assert that input is in the designated options (readability purposes only)
    if isinstance(input, str):
        input = input.lower()
    if input not in options:
        if argname is not None:
            raise ValueError(f"{argname} must be in {options}, received {input}")
        raise ValueError(f"input must be in {options}, received {input}")

def get_all_filepaths(path, filetype=None, contains=None):
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        all_files += [os.path.join(dirpath, file) for file in filenames]
        
    if filetype:
        if filetype.lower() != 'all':
            all_files = [file for file in all_files if file.endswith(filetype)]
    
    if contains:
        all_files = [file for file in all_files if contains in file]
        
    return all_files

def output_test_series(model, emulator_data, draws='random', k=10, save=True):
    test_scenarios = emulator_data.test_scenarios
    test_features = emulator_data.test_features
    test_labels = emulator_data.test_labels
    
    sectors = list(set(test_features.sectors))
    sectors.sort()

    if draws == 'random':
        data = random.sample(test_scenarios, k=k)
    elif draws == 'first':
        data = test_scenarios[:k]
    else:
        raise ValueError(f'draws must be in [random, first], received {draws}')
    
    for scen in data:
        single_scenario = scen
        test_model = single_scenario[0]
        test_exp = single_scenario[2]
        test_sector = single_scenario[1]
        single_test_features = torch.tensor(np.array(test_features[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64), dtype=torch.float)
        single_test_labels = np.array(test_labels[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64)
        preds = model.predict(single_test_features)

        plt.figure(figsize=(15,8))
        plt.plot(single_test_labels, 'r-', label='True')
        plt.plot(preds, 'b-', label='Predicted')
        plt.xlabel('Time (years since 2015)')
        plt.ylabel('SLE (mm)')
        plt.title(f'Model={test_model}, Exp={test_exp}, sector={sectors.index(test_sector)+1}')
        plt.legend()
        if save:
            plt.savefig(f'results/{test_model}_{test_exp}_test_sector.png')
            

def plot_true_vs_predicted(preds, y_test, save=None):
    try:
        preds = preds.detach().numpy()
    except AttributeError:
        pass
    plt.figure()
    plt.scatter(y_test, preds, s=3, alpha=0.2)
    plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], 'r-')
    plt.title('Neural Network True vs Predicted')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    if save:
        plt.savefig("results/nn.png")
    plt.show()
    

def _structure_emulatordata_args(input_args, time_series):
    emulator_data_defaults = dict(
                    target_column='sle',
                    drop_missing=True,
                    drop_columns=['groupname', 'experiment'],
                    boolean_indices=True,
                    scale=True,
                    split_type='batch_test',
                    drop_outliers={'column': 'ivaf', 'operator': '<', 'value': -1e13},
                    time_series=time_series,
                    lag=None
                    )
    
    if time_series:
        emulator_data_defaults['lag'] = 5
    
    # If no other args are supplied, use defaults
    if input_args is None:
        return emulator_data_defaults
    # else, replace provided key value pairs in the default dict and reassign
    else:
        for key in input_args.keys():
            emulator_data_defaults[key] = input_args[key]
        output_args = emulator_data_defaults
        

    
    return output_args

def _structure_architecture_args(architecture, time_series):
    if architecture is None and time_series:
        if 'nodes' in architecture.keys() or 'num_linear_layers' in architecture.keys():
            raise AttributeError(f'Time series architecture args must be in [num_rnn_layers, num_rnn_hidden], received {architecture}')
        architecture = {
            'num_rnn_layers': 3,
            'num_rnn_hidden': 128,
        }
    elif architecture is None and not time_series:
        if 'num_rnn_layers' in architecture.keys() or 'num_rnn_hidden' in architecture.keys():
            raise AttributeError(f'Time series architecture args must be in [num_linear_layers, nodes], received {architecture}')
        architecture = {
            'num_linear_layers': 4,
            'nodes': [128, 64, 32, 1],
        }
    else:
        return architecture
    return architecture

def load_ml_data(data_directory, time_series):
    if time_series:
        try:
            test_features = pd.read_csv(f'{data_directory}/ts_test_features.csv')
            train_features = pd.read_csv(f'{data_directory}/ts_train_features.csv')
            test_labels = pd.read_csv(f'{data_directory}/ts_test_labels.csv')
            train_labels = pd.read_csv(f'{data_directory}/ts_train_labels.csv')
        except FileNotFoundError:
                raise FileNotFoundError(f'Files not found at {data_directory}. Format must be in format \"ts_train_features.csv\"')
    else:
        try:
            test_features = pd.read_csv(f'{data_directory}/traditional_test_features.csv')
            train_features = pd.read_csv(f'{data_directory}/traditional_train_features.csv')
            test_labels = pd.read_csv(f'{data_directory}/traditional_test_labels.csv')
            train_labels = pd.read_csv(f'{data_directory}/traditional_train_labels.csv')
        except FileNotFoundError:
                raise FileNotFoundError(f'Files not found at {data_directory}. Format must be in format \"traditional_train_features.csv\"')
    
    return train_features, train_labels, test_features, test_labels

