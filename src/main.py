from data.processing.aggregate_by_sector import aggregate_atmosphere, aggregate_icecollapse, aggregate_ocean
from data.processing.process_outputs import process_repository
from data.processing.combine_datasets import combine_datasets
from data.classes.EmulatorData import EmulatorData
from models.SimpleEmulator import SimpleEmulator
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





emulator_data = EmulatorData(directory=export_dir)
emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
    target_column='ivaf',
    drop_missing=True,
    drop_columns=False,
    boolean_indices=True,
    scale=True,
    split_type='batch'
)



class EmulatorTrainingDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
    
X_train = np.array(train_features, dtype=np.float64)
y_train = np.array(train_labels, dtype=np.float64)
X_test = np.array(test_features, dtype=np.float64)
y_test = np.array(test_labels, dtype=np.float64)

train_dataset = EmulatorTrainingDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().squeeze())
train_loader = DataLoader(dataset=train_dataset, batch_size=100)

model = Emulator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),)
epochs = 100
verbose = True
loss_list = []
batch_losses = []
for epoch in range(1, epochs+1):

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
        batch_losses.append(loss.item())
            
    # loss_list.append(loss)
    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)

    if epoch % 5 == 0:
        print(f"Epoch: {epoch}, MSE: {avg_loss}")
        
model.eval()
X_test = torch.tensor(X_test, dtype=torch.float)
preds = model(X_test)

print('')
print(f"""--- PyTorch Test Metrics---
Mean Absolute Error: {mean_absolute_error(y_test, preds.detach().numpy())}
Mean Squared Error: {mean_squared_error(y_test, preds.detach().numpy())}
R2 Score: {r2_score(y_test, preds.detach().numpy())}""")

plt.figure()
plt.scatter(y_test, preds.detach().numpy(), s=3)
plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], 'r-')
plt.title('Neural Network True vs Predicted')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.savefig("nn.png")
plt.show()

plt.figure()
plt.plot(loss_list, '-')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss (MSE)')
plt.savefig('epoch_loss.png')
plt.show()

plt.figure()
plt.plot(batch_losses, 'r-')
plt.title('Loss per Batch')
plt.xlabel('Batch')
plt.ylabel('Loss (MSE)')
# plt.ylim([0,0.1])
plt.savefig('batch_loss.png')
plt.show()

stop = ''