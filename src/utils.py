import yaml
import os
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

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


def check_input(input, options):
    # simple assert that input is in the designated options (readability purposes only)
    if isinstance(input, str):
        input = input.lower()
    assert input in options, f"input must be in {options}, received {input}"

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