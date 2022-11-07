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



# model = tf.keras.Sequential([
# #       normalizer,
#       tf.keras.layers.Dense(128, activation='relu'),
#       tf.keras.layers.Dense(64, activation='relu'),
#       tf.keras.layers.Dense(1)
#   ])

# model.compile(loss='mean_squared_error',
#                 optimizer=tf.keras.optimizers.Adam())

# history = model.fit(
#     train_features,
#     train_labels,
#     shuffle=True,
#     validation_split=0.2,
#     verbose=2, epochs=20)

# def plot_loss(history):
#     plt.plot(history.history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error [MSE]')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(r'results/training.png')
    
# plot_loss(history)

# model.evaluate(test_features, test_labels)

# preds = model.predict(test_features)

# print(f"""--- Tensorflow Test ---
# Mean Absolute Error: {mean_absolute_error(test_labels, preds)}
# Mean Squared Error: {mean_squared_error(test_labels, preds)}
# R2 Score: {r2_score(test_labels, preds)}""")
    
# plt.figure()
# plt.plot(preds, test_labels, 'o')
# plt.savefig(r'results/nn.png')

# if split_type == 'batch':
#     for scen in emulator_data.test_scenarios[:10]:
#         single_scenario = scen
#         test_model = single_scenario[0]
#         test_exp = single_scenario[2]
#         test_sector = single_scenario[1]
#         single_test_features = np.array(test_features[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64)
#         single_test_labels = np.array(test_labels[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64)
#         preds = model(single_test_features)

#         single_test_labels = emulator_data.unscale(single_test_labels.reshape(-1,1), 'outputs') * 1e-9 / 361.8
#         preds = emulator_data.unscale(np.array(preds).reshape(-1,1), 'outputs') * 1e-9 / 361.8

#         plt.figure()
#         plt.plot(single_test_labels, 'r-', label='True')
#         plt.plot(preds, 'b-', label='Predicted')
#         plt.xlabel('Time (years since 2015)')
#         plt.ylabel('SLE (mm)')
#         plt.title(f'Model={test_model}, Exp={test_exp}')
#         plt.ylim([-10,10])
#         plt.legend()
#         plt.savefig(f'results/{test_model}_{test_exp}_{round(test_sector)}.png')

# X_train = torch.from_numpy(train_features).float()
# y_train = torch.from_numpy(train_labels).float()
# X_test = torch.from_numpy(test_features).float()
# y_test = torch.from_numpy(test_labels).float()


print('3/4: Training Model')

data_dict = {'train_features': train_features,
             'train_labels': train_labels,
             'test_features': test_features,
             'test_labels': test_labels,  }

trainer = Trainer(cfg)
trainer.train(
    model=ExploratoryModel.ExploratoryModel, 
    data_dict=data_dict, 
    criterion=nn.MSELoss(), 
    epochs=100, 
    batch_size=200,
    tensorboard=True,
    num_linear_layers=4,
    nodes=[256, 128, 64, 1],
)

print('4/4: Evaluating Model')

model = trainer.model
metrics, preds = trainer.evaluate()


y_test = trainer.y_test
plt.figure()
plt.scatter(y_test, preds.detach().numpy(), s=3, alpha=0.2)
plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], 'r-')
plt.title('Neural Network True vs Predicted')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.savefig("results/nn.png")
plt.show()

# TODO: Plot validation
# TODO: Try other metrics / tensorboard

for scen in emulator_data.test_scenarios[:10]:
    single_scenario = scen
    test_model = single_scenario[0]
    test_exp = single_scenario[2]
    test_sector = single_scenario[1]
    single_test_features = torch.tensor(np.array(test_features[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64), dtype=torch.float)
    single_test_labels = np.array(test_labels[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64)
    preds = model(single_test_features).detach().numpy()

    single_test_labels = emulator_data.unscale(single_test_labels.reshape(-1,1), 'outputs') * 1e-9 / 361.8
    preds = emulator_data.unscale(preds.reshape(-1,1), 'outputs') * 1e-9 / 361.8

    plt.figure()
    plt.plot(single_test_labels, 'r-', label='True')
    plt.plot(preds, 'b-', label='Predicted')
    plt.xlabel('Time (years since 2015)')
    plt.ylabel('SLE (mm)')
    plt.title(f'Model={test_model}, Exp={test_exp}')
    plt.ylim([-10,10])
    plt.legend()
    plt.savefig(f'results/{1}_{test_model}_{test_exp}.png')

stop = ''