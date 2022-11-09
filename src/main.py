from data.processing.aggregate_by_sector import aggregate_atmosphere, aggregate_icecollapse, aggregate_ocean
from data.processing.process_outputs import process_repository
from data.processing.combine_datasets import combine_datasets
from data.classes.EmulatorData import EmulatorData
from training.Trainer import Trainer
from models import FC3_N128, FC6_N256, FC12_N1024, ExploratoryModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import get_configs
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from tqdm import tqdm
# from sklearn.preprocessing import MinMaxScaler
cfg = get_configs()


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
split_type = 'batch'

print('2/4: Processing Data')
emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
    target_column='ivaf',
    drop_missing=True,
    drop_columns=False,
    boolean_indices=True,
    scale=True,
    split_type='batch_test'
)

print('3/4: Training Model')

# emulator_data.unscale(test_features)
emulator_data.unscale(test_labels, 'outputs')

data_dict = {'train_features': train_features,
             'train_labels': train_labels,
             'test_features': test_features,
             'test_labels': test_labels,  }



models = {
    'normal': {
        'num_linear_layers': 6,
        'nodes': [256, 128, 64, 32, 16, 1],
        },
    
    'smaller': {
        'num_linear_layers': 4,
        'nodes': [128, 64, 32, 1]
        },
        
    'smallest': {
        'num_linear_layers': 3,
        'nodes': [64, 20, 1]
    },
    
    'largest': {
        'num_linear_layers': 8,
        'nodes': [256, 128, 64, 32, 16, 8, 4, 1]
    },
    
    'normal_expanding': {
        'num_linear_layers': 6,
        'nodes': [64, 128, 32, 16, 8, 1]
    },
}

count = 0
for iteration in range(5):
    for batch_size in [50, 100, 250]:
        for run, settings in models.items():
            print('')
            print(f"Training... Model: {run}, Batch Size: {batch_size}, Iteration: {iteration}, Trained {count} models")
            trainer = Trainer(cfg)
            trainer.train(
                model=ExploratoryModel.ExploratoryModel, 
                num_linear_layers=settings['num_linear_layers'],
                nodes=settings['nodes'],
                data_dict=data_dict, 
                criterion=nn.MSELoss(), 
                epochs=100, 
                batch_size=batch_size,
                tensorboard=True,
                save_model=False,
            )
            
            count += 1



# trainer = Trainer(cfg)
# trainer.train(
#     model=ExploratoryModel.ExploratoryModel, 
#     num_linear_layers=6,
#     nodes=[256, 128, 64, 32, 16, 1],
#     data_dict=data_dict, 
#     criterion=nn.MSELoss(), 
#     epochs=200, 
#     batch_size=200,
#     tensorboard=True,
#     save_model=True,
# )


# print('4/4: Evaluating Model')

# model = trainer.model
# metrics, preds = trainer.evaluate()


# y_test = trainer.y_test
# plt.figure()
# plt.scatter(y_test, preds.detach().numpy(), s=3, alpha=0.2)
# plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], 'r-')
# plt.title('Neural Network True vs Predicted')
# plt.xlabel('True')
# plt.ylabel('Predicted')
# plt.savefig("results/nn.png")
# plt.show()

# # TODO: Plot validation
# # TODO: Try other metrics / tensorboard

# for scen in emulator_data.test_scenarios[:10]:
#     single_scenario = scen
#     test_model = single_scenario[0]
#     test_exp = single_scenario[2]
#     test_sector = single_scenario[1]
#     single_test_features = torch.tensor(np.array(test_features[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64), dtype=torch.float)
#     single_test_labels = np.array(test_labels[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64)
#     preds = model(single_test_features).detach().numpy()

#     single_test_labels = emulator_data.unscale(single_test_labels.reshape(-1,1), 'outputs') * 1e-9 / 361.8
#     preds = emulator_data.unscale(preds.reshape(-1,1), 'outputs') * 1e-9 / 361.8

#     plt.figure()
#     plt.plot(single_test_labels, 'r-', label='True')
#     plt.plot(preds, 'b-', label='Predicted')
#     plt.xlabel('Time (years since 2015)')
#     plt.ylabel('SLE (mm)')
#     plt.title(f'Model={test_model}, Exp={test_exp}')
#     plt.ylim([-10,10])
#     plt.legend()
#     plt.savefig(f'results/{1}_{test_model}_{test_exp}.png')

# stop = ''