from ise.data.EmulatorData import EmulatorData
from ise.models.training.Trainer import Trainer
from ise.models.timeseries.TimeSeriesEmulator import TimeSeriesEmulator
from ise.models.traditional.ExploratoryModel import ExploratoryModel
from datetime import datetime
from torch import nn

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

def lag_sequence_test(data_directory, lag_array, sequence_array, iterations, model=TimeSeriesEmulator,
                      emulator_data_args=None, architecture=None, verbose=True, epochs=100,
                      batch_size=100, loss=nn.MSELoss(), ):
    
    if verbose:
        print('1/3: Loading processed data...')
        
    emulator_data_args = _structure_emulatordata_args(emulator_data_args, time_series=True)
    architecture = _structure_architecture_args(architecture, time_series=True)

    count = 0
    for iteration in range(1, iterations+1):
        for lag in lag_array:
            for sequence_length in sequence_array:
                
                print(f"Training... Lag: {lag}, Sequence Length: {sequence_length}, Iteration: {iteration}, Trained {count} models")
                
                emulator_data = EmulatorData(directory=data_directory)
                emulator_data_args['lag'] = lag
                emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
                    **emulator_data_args
                )

                data_dict = {'train_features': train_features,
                            'train_labels': train_labels,
                            'test_features': test_features,
                            'test_labels': test_labels, }
                
                trainer = Trainer()
                
                current_time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")
                trainer.train(
                    model=model,
                    architecture=architecture,
                    data_dict=data_dict,
                    criterion=loss,
                    epochs=epochs,
                    batch_size=batch_size,
                    tensorboard=True,
                    save_model=False,
                    performance_optimized=False,
                    verbose=verbose,
                    sequence_length=sequence_length,
                    tensorboard_comment=f" -- {current_time}, lag={lag}, sequence_length={sequence_length}"
                )
                
                # not verbose because if verbose==True, this is already calculated in training loop
                if not verbose:
                    metrics, preds = trainer.evaluate()
                
                count += 1
    
    print(f'Finished trainin {count} models.')
    
    
def rnn_architecture_test(data_directory, rnn_layers_array, hidden_nodes_array, iterations,
                          model=TimeSeriesEmulator, emulator_data_args=None, architecture=None, verbose=True, 
                          epochs=100, batch_size=100, loss=nn.MSELoss(), ):
    
    
    emulator_data_args = _structure_emulatordata_args(input_args=emulator_data_args, time_series=True)
    
    emulator_data = EmulatorData(directory=data_directory)
    emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
        **emulator_data_args
    )
    
    data_dict = {'train_features': train_features,
                'train_labels': train_labels,
                'test_features': test_features,
                'test_labels': test_labels, }      
    
    count = 0
    for iteration in range(1, iterations+1):
        for num_rnn_layers in rnn_layers_array:
            for num_rnn_hidden in hidden_nodes_array:
                print(f"Training... RNN Layers: {num_rnn_layers}, Hidden: {num_rnn_hidden}, Iteration: {iteration}, Trained {count} models")
            
                trainer = Trainer()
                architecture = {
                    'num_rnn_layers': num_rnn_layers,
                    'num_rnn_hidden': num_rnn_hidden,
                }
                current_time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")
                trainer.train(
                    model=model,
                    architecture=architecture,
                    data_dict=data_dict,
                    criterion=nn.MSELoss(),
                    epochs=epochs,
                    batch_size=batch_size,
                    tensorboard=True,
                    save_model=True,
                    performance_optimized=False,
                    verbose=verbose,
                    sequence_length=5,
                    tensorboard_comment=f" -- {current_time}, num_rnn={num_rnn_layers}, num_hidden={num_rnn_hidden}"
                )
                
                if not verbose:
                    metrics, preds = trainer.evaluate()
                
                count += 1
    
    print(f'Finished trainin {count} models.')
    
    
def traditional_architecture_test(data_directory, architectures: list[dict], iterations,
                          model=ExploratoryModel, emulator_data_args=None,  verbose=True, 
                          epochs=100, batch_size=100, loss=nn.MSELoss(), ):
    
    
    emulator_data_args = _structure_emulatordata_args(input_args=emulator_data_args, time_series=False)
    
    emulator_data = EmulatorData(directory=data_directory)
    emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
        **emulator_data_args
    )
    
    data_dict = {'train_features': train_features,
                'train_labels': train_labels,
                'test_features': test_features,
                'test_labels': test_labels, }      
    
    count = 0
    for iteration in range(1, iterations+1):
        for architecture in architectures:
            num_linear_layers = architecture['num_linear_layers']
            nodes = architecture['nodes']
            print(f"Training... Linear Layers: {num_linear_layers}, Nodes: {nodes}, Iteration: {iteration}, Trained {count} models")
        
            trainer = Trainer()
            current_time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")
            trainer.train(
                model=model,
                architecture=architecture,
                data_dict=data_dict,
                criterion=nn.MSELoss(),
                epochs=epochs,
                batch_size=batch_size,
                tensorboard=True,
                save_model=True,
                performance_optimized=False,
                verbose=verbose,
                sequence_length=5,
                tensorboard_comment=f" -- {current_time}, num_linear={num_linear_layers}, nodes={nodes}"
            )
            
            if not verbose:
                metrics, preds = trainer.evaluate()
            
            count += 1
    
    print(f'Finished trainin {count} models.')
    

# TODO: Write tests for the above


# TODO: make dense_architecture_test() that tests what dense layers should be at the end of the RNN


# TODO: Write function for dataset_tests and other tests I've done before (reproducibility!!!), use dict{'dataset1':['columns']} to loop

# dataset1 = ['mrro_anomaly', 'rhoi', 'rhow', 'groupname', 'experiment']
# dataset2 = ['mrro_anomaly', 'rhoi', 'rhow', 'groupname', 'experiment', 'ice_shelf_fracture', 'tier', ]
# dataset3 = ['mrro_anomaly', 'groupname', 'experiment']
# dataset4 = ['groupname', 'experiment']
# dataset5 = ['groupname', 'experiment', 'regions', 'tier']

# def dataset_tests(datasets)
# count = 0
# for iteration in range(5):
#     for dataset in ['dataset5']:
#         print('')
#         print(f"Training... Dataset: {dataset}, Iteration: {iteration}, Trained {count} models")
#         test_features = pd.read_csv(f'/users/pvankatw/emulator/src/data/ml/{dataset}/test_features.csv')
#         train_features = pd.read_csv(f'/users/pvankatw/emulator/src/data/ml/{dataset}/train_features.csv')
#         test_labels = pd.read_csv(f'/users/pvankatw/emulator/src/data/ml/{dataset}/test_labels.csv')
#         train_labels = pd.read_csv(f'/users/pvankatw/emulator/src/data/ml/{dataset}/train_labels.csv')
#         scenarios = pd.read_csv(f'/users/pvankatw/emulator/src/data/ml/{dataset}/test_scenarios.csv').values.tolist()
#
#
#         data_dict = {'train_features': train_features,
#                     'train_labels': train_labels,
#                     'test_features': test_features,
#                     'test_labels': test_labels,  }
#
#         start = time.time()
#         trainer = Trainer()
#         trainer.train(
#             model=ExploratoryModel.ExploratoryModel,
#             num_linear_layers=6,
#             nodes=[256, 128, 64, 32, 16, 1],
#             # num_linear_layers=12,
#             # nodes=[2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
#             data_dict=data_dict,
#             criterion=nn.MSELoss(),
#             epochs=100,
#             batch_size=100,
#             tensorboard=True,
#             save_model=True,
#             performance_optimized=False,
#         )
#         print(f'Total Time: {time.time() - start:0.4f} seconds')
#
#         print('4/4: Evaluating Model')
#
#         model = trainer.model
#         metrics, preds = trainer.evaluate()
#
#         count += 1