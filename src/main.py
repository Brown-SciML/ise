from data.processing.aggregate_by_sector import aggregate_atmosphere, aggregate_icecollapse, aggregate_ocean
from data.processing.process_outputs import process_repository
from data.processing.combine_datasets import combine_datasets
from data.classes.EmulatorData import EmulatorData
from models import FC3_N128, FC6_N256, FC12_N1024
from training.PyTorchDataset import PyTorchDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import get_configs
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
cfg = get_configs()
import pandas as pd
import time
import tensorflow as tf

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
    split_type='batch'
)
    
X_train = np.array(train_features, dtype=np.float64)
y_train = np.array(train_labels, dtype=np.float64)
X_test = np.array(test_features, dtype=np.float64)
y_test = np.array(test_labels, dtype=np.float64)

train_dataset = PyTorchDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().squeeze())
test_dataset = PyTorchDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float().squeeze())
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=50,)


print('3/4: Training Model')
# model = FC3_N128.FC3_N128(input_layer_size=X_train.shape[1])
model = FC6_N256.FC6_N256(input_layer_size=X_train.shape[1])
# model = FC12_N1024.FC12_N1024(input_layer_size=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),)
epochs = 20
verbose = True
logs = {'training': {'epoch': [], 'batch': []}, 'testing': []}
batch_losses = []
for epoch in range(1, epochs+1):
    epoch_start = time.time()
    model.train()
    iteration = 0
    total_loss = 0
    for X_train_batch, y_train_batch in train_loader:
        # Train model
        optimizer.zero_grad()

        pred = model(X_train_batch)
        loss = criterion(pred, y_train_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        logs['training']['batch'].append(loss.item())
    
    training_end = time.time()
    model.eval()
    test_total_loss = 0
    for X_test_batch, y_test_batch in test_loader:
        test_pred = model(X_test_batch)
        loss = criterion(test_pred, y_test_batch.unsqueeze(1))
        test_total_loss += loss.item()
        
    test_loss = test_total_loss / len(test_loader)
    logs['testing'].append(test_loss)
    
    
    testing_end = time.time()
    avg_loss = total_loss / len(train_loader)
    logs['training']['epoch'].append(avg_loss)

    if epoch % 1 == 0:
        print('')
        print(f"""Epoch: {epoch}/{epochs}, Training Loss (MSE): {avg_loss:0.8f}, Validation Loss (MSE): {test_loss:0.8f}
Training time: {training_end - epoch_start: 0.2f} seconds, Validation time: {testing_end - training_end: 0.2f} seconds""")

print('4/4: Testing & Plotting')
model.eval()
X_test = torch.tensor(X_test, dtype=torch.float)
preds = model(X_test)

print('')
print(f"""--- PyTorch Test Metrics---
Mean Absolute Error: {mean_absolute_error(y_test, preds.detach().numpy())}
Mean Squared Error: {mean_squared_error(y_test, preds.detach().numpy())}
R2 Score: {r2_score(y_test, preds.detach().numpy())}""")

plt.figure()
plt.scatter(y_test, preds.detach().numpy(), s=3, alpha=0.2)
plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], 'r-')
plt.title('Neural Network True vs Predicted')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.savefig("results/nn.png")
plt.show()

plt.figure()
plt.plot(logs['training']['epoch'], 'r-', label='Training')
plt.plot(logs['testing'], 'b-', label='Validation')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss (MSE)')
# plt.ylim([0,0.005])
plt.legend()
plt.savefig('results/epoch_loss.png')
plt.show()

plt.figure()
plt.plot(logs['training']['batch'], 'r-')
plt.title('Loss per Batch')
plt.xlabel('Batch')
plt.ylabel('Loss (MSE)')
# plt.ylim([0,0.1])
plt.savefig('results/batch_loss.png')
plt.show()
# TODO: Plot validation
# TODO: Try other metrics / tensorboard

if split_type == 'batch':
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
        plt.savefig(f'results/{test_model}_{test_exp}_{round(test_sector)}.png')

stop = ''