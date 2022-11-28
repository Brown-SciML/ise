from data.processing.aggregate_by_sector import aggregate_atmosphere, aggregate_icecollapse, aggregate_ocean
from data.processing.process_outputs import process_repository
from data.processing.combine_datasets import combine_datasets
from data.classes.EmulatorData import EmulatorData
from training.Trainer import Trainer
from models import ExploratoryModel, TimeSeriesEmulator
from utils import get_configs
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime

cfg = get_configs()

np.random.seed(10)

forcing_directory = cfg['data']['forcing']
zenodo_directory = cfg['data']['output']
export_dir = cfg['data']['export']
processing = cfg['processing']
data_directory = cfg['data']['directory']

if processing['generate_atmospheric_forcing']:
    af_directory = f"{forcing_directory}/Atmosphere_Forcing/"
    # TODO: refactor model_in_columns as aogcm_as_features
    aggregate_atmosphere(af_directory, export=export_dir, model_in_columns=False, )

if processing['generate_oceanic_forcing']:
    of_directory = f"{forcing_directory}/Ocean_Forcing/"
    aggregate_ocean(of_directory, export=export_dir, model_in_columns=False, )

if processing['generate_icecollapse_forcing']:
    ice_directory = f"{forcing_directory}/Ice_Shelf_Fracture"
    aggregate_icecollapse(ice_directory, export=export_dir, model_in_columns=False, )

if processing['generate_outputs']:
    outputs = process_repository(zenodo_directory, export_filepath=f"{export_dir}/outputs.csv")

if processing['combine_datasets']:
    master, inputs, outputs = combine_datasets(processed_data_dir=export_dir,
                                               include_icecollapse=processing['include_icecollapse'],
                                               export=export_dir)

print('1/4: Loading in Data')
emulator_data = EmulatorData(directory=export_dir)
print('2/4: Processing Data')
lag = 5
emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
    target_column='sle',
    drop_missing=True,
    drop_columns=['groupname', 'experiment'],
    boolean_indices=True,
    scale=True,
    split_type='batch_test',
    drop_outliers={'column': 'ivaf', 'operator': '<', 'value': -1e13},
    time_series=True,
    lag=lag,
)

data_dict = {'train_features': train_features,
             'train_labels': train_labels,
             'test_features': test_features,
             'test_labels': test_labels, }
trainer = Trainer(cfg)

print('3/4: Training Model')
exploratory_architecture = {
    'num_linear_layers': 6,
    'nodes': [256, 128, 64, 32, 16, 1],
}
time_series_architecture = {
    'num_rnn_layers': 3,
    'num_rnn_hidden': 128,
}

trainer.train(
    model=TimeSeriesEmulator.TimeSeriesEmulator,
    architecture=time_series_architecture,
    data_dict=data_dict,
    criterion=nn.MSELoss(),
    epochs=20,
    batch_size=100,
    tensorboard=False,
    save_model=False,
    performance_optimized=True,
)
print('4/4: Evaluating Model')
model = trainer.model
metrics, preds = trainer.evaluate()


print(metrics)

# dataset = 'dataset5'
# test_features = pd.read_csv(f'./data/ml/{dataset}/test_features.csv')
# train_features = pd.read_csv(f'./data/ml/{dataset}/train_features.csv')
# test_labels = pd.read_csv(f'./data/ml/{dataset}/test_labels.csv')
# train_labels = pd.read_csv(f'./data/ml/{dataset}/train_labels.csv')
# scenarios = pd.read_csv(f'./data/ml/{dataset}/test_scenarios.csv').values.tolist()



# dataset1 = ['mrro_anomaly', 'rhoi', 'rhow', 'groupname', 'experiment']
# dataset2 = ['mrro_anomaly', 'rhoi', 'rhow', 'groupname', 'experiment', 'ice_shelf_fracture', 'tier', ]
# dataset3 = ['mrro_anomaly', 'groupname', 'experiment']
# dataset4 = ['groupname', 'experiment']
# dataset5 = ['groupname', 'experiment', 'regions', 'tier']

# print('1/4: Loading in Data')
# emulator_data = EmulatorData(directory=export_dir)
# split_type = 'batch'

# print('2/4: Processing Data')
# emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
#     target_column='sle',
#     drop_missing=True,
#     drop_columns=dataset5,
#     # drop_columns=False,
#     boolean_indices=True,
#     scale=True,
#     split_type='batch_test',
#     drop_outliers={'column': 'ivaf', 'operator': '<', 'value': -1e13}
# )

# import pandas as pd
# train_features.to_csv(r'/users/pvankatw/emulator/src/data/ml/dataset5/train_features.csv', index=False)
# test_features.to_csv(r'/users/pvankatw/emulator/src/data/ml/dataset5/test_features.csv', index=False)
# pd.Series(train_labels, name='sle').to_csv(r'/users/pvankatw/emulator/src/data/ml/dataset5/train_labels.csv', index=False)
# pd.Series(test_labels, name='sle').to_csv(r'/users/pvankatw/emulator/src/data/ml/dataset5/test_labels.csv', index=False)
# pd.DataFrame(emulator_data.test_scenarios).to_csv(r'/users/pvankatw/emulator/src/data/ml/dataset5/test_scenarios.csv', index=False)

# TODO: Write function for dataset_tests and other tests I've done before (reproducibility!!!), use dict{'dataset1':['columns']} to loop
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
#         trainer = Trainer(cfg)
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


try:
    preds = preds.detach().numpy()
except AttributeError:
    pass


y_test = trainer.y_test
plt.figure()
plt.scatter(y_test, preds, s=3, alpha=0.2)
plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], 'r-')
plt.title('Neural Network True vs Predicted')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.savefig("results/nn.png")
plt.show()


def output_test_series(test_scenarios, draws='random', k=10, save=True):
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
    
    
def lag_sequence_test(lag_array, sequence_array, iterations):
    count = 0
    for iteration in range(1, iterations+1):
        for lag in lag_array:
            for sequence_length in sequence_array:
                print(f"Training... Lag: {lag}, Sequence Length: {sequence_length}, Iteration: {iteration}, Trained {count} models")
                emulator_data = EmulatorData(directory=export_dir)
                emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
                    target_column='sle',
                    drop_missing=True,
                    drop_columns=['groupname', 'experiment'],
                    boolean_indices=True,
                    scale=True,
                    split_type='batch_test',
                    drop_outliers={'column': 'ivaf', 'operator': '<', 'value': -1e13},
                    time_series=True,
                    lag=lag,
                )

                data_dict = {'train_features': train_features,
                            'train_labels': train_labels,
                            'test_features': test_features,
                            'test_labels': test_labels, }
                trainer = Trainer(cfg)
                time_series_architecture = {
                    'num_rnn_layers': 3,
                    'num_rnn_hidden': 128,
                }
                current_time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")
                trainer.train(
                    model=TimeSeriesEmulator.TimeSeriesEmulator,
                    architecture=time_series_architecture,
                    data_dict=data_dict,
                    criterion=nn.MSELoss(),
                    epochs=100,
                    batch_size=100,
                    tensorboard=True,
                    save_model=False,
                    performance_optimized=False,
                    verbose=False,
                    sequence_length=sequence_length,
                    tensorboard_comment=f" -- {current_time}, lag={lag}, sequence_length={sequence_length}"
                )
                metrics, preds = trainer.evaluate()
                print('Metrics:', metrics)
                
                count += 1
                

def architecture_test(rnn_layers_array, hidden_nodes_array, linear_layers_array, iterations):
    count = 0
    for iteration in range(1, iterations+1):
        for num_rnn_layers in rnn_layers_array:
            for num_rnn_hidden in hidden_nodes_array:
                print(f"Training... RNN Layers: {num_rnn_layers}, Hidden: {num_rnn_hidden}, Iteration: {iteration}, Trained {count} models")
                emulator_data = EmulatorData(directory=export_dir)
                emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
                    target_column='sle',
                    drop_missing=True,
                    drop_columns=['groupname', 'experiment'],
                    boolean_indices=True,
                    scale=True,
                    split_type='batch_test',
                    drop_outliers={'column': 'ivaf', 'operator': '<', 'value': -1e13},
                    time_series=True,
                    lag=lag,
                )

                data_dict = {'train_features': train_features,
                            'train_labels': train_labels,
                            'test_features': test_features,
                            'test_labels': test_labels, }
                trainer = Trainer(cfg)
                time_series_architecture = {
                    'num_rnn_layers': 3,
                    'num_rnn_hidden': 128,
                }
                current_time = datetime.now().strftime(r"%d-%m-%Y %H.%M.%S")
                trainer.train(
                    model=TimeSeriesEmulator.TimeSeriesEmulator,
                    architecture=time_series_architecture,
                    data_dict=data_dict,
                    criterion=nn.MSELoss(),
                    epochs=100,
                    batch_size=100,
                    tensorboard=True,
                    save_model=False,
                    performance_optimized=False,
                    verbose=False,
                    sequence_length=10, # update with results from lag_sequence_test
                    tensorboard_comment=f" -- {current_time}, num_rnn={num_rnn_layers}, num_hidden={num_rnn_hidden}"
                )
                metrics, preds = trainer.evaluate()
                print('Metrics:', metrics)
                
                count += 1
    

stop = ''
