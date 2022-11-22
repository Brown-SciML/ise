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
import time
import pandas as pd

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
emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
    target_column='sle',
    drop_missing=True,
    drop_columns=['groupname', 'experiment'],
    boolean_indices=True,
    scale=True,
    split_type='batch_test',
    drop_outliers={'column': 'ivaf', 'operator': '<', 'value': -1e13},
    time_series=True
)

data_dict = {'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels,  }
trainer = Trainer(cfg)

print('3/4: Training Model')
architecture = {''}
trainer.train(
    model=TimeSeriesEmulator.TimeSeriesEmulator,
    num_linear_layers=6,
    nodes=[256, 128, 64, 32, 16, 1],
    data_dict=data_dict,
    criterion=nn.MSELoss(),
    epochs=100,
    batch_size=100,
    tensorboard=True,
    save_model=True,
    performance_optimized=False,
)
print('4/4: Evaluating Model')
model = trainer.model
metrics, preds = trainer.evaluate()

# dataset = 'dataset5'
# test_features = pd.read_csv(f'./data/ml/{dataset}/test_features.csv')
# train_features = pd.read_csv(f'./data/ml/{dataset}/train_features.csv')
# test_labels = pd.read_csv(f'./data/ml/{dataset}/test_labels.csv')
# train_labels = pd.read_csv(f'./data/ml/{dataset}/train_labels.csv')
# scenarios = pd.read_csv(f'./data/ml/{dataset}/test_scenarios.csv').values.tolist()




train_features = train_features





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




# try:
#     preds = preds.detach().numpy()
# except AttributeError:
#     pass


# y_test = trainer.y_test
# plt.figure()
# plt.scatter(y_test, preds, s=3, alpha=0.2)
# plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], 'r-')
# plt.title('Neural Network True vs Predicted')
# plt.xlabel('True')
# plt.ylabel('Predicted')
# plt.savefig("results/nn.png")
# plt.show()

# # TODO: Plot validation
# # TODO: Try other metrics / tensorboard

# for scen in scenarios[:10]:
#     single_scenario = scen
#     test_model = single_scenario[0]
#     test_exp = single_scenario[2]
#     test_sector = single_scenario[1]
#     single_test_features = torch.tensor(np.array(test_features[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64), dtype=torch.float).to(trainer.device)
#     single_test_labels = np.array(test_labels[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64)
#     preds = model(single_test_features).cpu().detach().numpy()

#     # single_test_labels = emulator_data.unscale(single_test_labels.reshape(-1,1), 'outputs') * 1e-9 / 361.8
#     # preds = emulator_data.unscale(preds.reshape(-1,1), 'outputs') * 1e-9 / 361.8

#     plt.figure()
#     plt.plot(single_test_labels, 'r-', label='True')
#     plt.plot(preds, 'b-', label='Predicted')
#     plt.xlabel('Time (years since 2015)')
#     plt.ylabel('SLE (mm)')
#     plt.title(f'Model={test_model}, Exp={test_exp}')
#     plt.ylim([-10,10])
#     plt.legend()
#     # plt.savefig(f'results/{1}_{test_model}_{test_exp}.png')

stop = ''