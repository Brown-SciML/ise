import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(10)


file_dir = os.path.dirname(os.path.realpath(__file__))

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
    """Formats the arguments for model architectures.

    Args:
        architecture (dict): User input for desired architecture.
        time_series (bool): Flag denoting whether to use time series model arguments or traditional.

    Returns:
        architecture (dict): Formatted architecture argument.
    """
    
    # Check to make sure inappropriate args are not used
    if not time_series and ('num_rnn_layers' in architecture.keys() or 'num_rnn_hidden' in architecture.keys()):
            raise AttributeError(f'Time series architecture args must be in [num_linear_layers, nodes], received {architecture}')
    if time_series and ('nodes' in architecture.keys() or 'num_linear_layers' in architecture.keys()):
            raise AttributeError(f'Time series architecture args must be in [num_rnn_layers, num_rnn_hidden], received {architecture}')
        
    if architecture is None:
        if time_series:
            architecture = {
                'num_rnn_layers': 3,
                'num_rnn_hidden': 128,
            }
        else:
            architecture = {
                'num_linear_layers': 4,
                'nodes': [128, 64, 32, 1],
            }
    else:
        return architecture
    return architecture



