from data.processing.aggregate_by_sector import aggregate_atmosphere, aggregate_icecollapse, aggregate_ocean
from data.processing.process_outputs import process_repository
from data.processing.combine_datasets import combine_datasets
from data.classes.EmulatorData import EmulatorData
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
    

# TODO: Start using my EmulatorData class -- find out what step is causing problems
emulator_data = EmulatorData(directory=export_dir)
# data = pd.read_csv(r"/users/pvankatw/emulator/src/data/files/master.csv", low_memory=False)
data = emulator_data.drop_missing().drop_columns(columns=['experiment', 'exp_id', 'groupname', 'regions'])
# dataset = data.dropna().drop(columns=['experiment', 'exp_id', 'groupname', 'regions'])
dataset = pd.get_dummies(data.data)

for col in dataset.columns:
    dataset[col] = dataset[col].astype(float)
    
scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = pd.DataFrame(scaler.transform(dataset), columns=dataset.columns)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

output_cols = ['icearea','iareafl','iareagr','ivol','ivaf','smb','smbgr', 'bmbfl']
train_features = train_dataset.drop(columns=output_cols)
test_features = test_dataset.drop(columns=output_cols)
train_labels = train_dataset['ivaf']
test_labels = test_dataset['ivaf']

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_layer_size = X_train.shape[1]
        self.layer = torch.nn.Linear(self.input_layer_size, 128)
        self.layer2 = torch.nn.Linear(128, 32)
        self.layer3 = torch.nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)     
        return x

class EmulatorDataset(Dataset):
    
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
train_dataset = EmulatorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
test_dataset = EmulatorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

train_loader = DataLoader(dataset=train_dataset, batch_size=100)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),)
epochs = 100
verbose = True
loss_list = []
for epoch in range(epochs):
    if verbose and epoch % 1 == 0:
        print(f'---------- EPOCH: {epoch} ----------')

    iteration = 0
    for X_train_batch, y_train_batch in train_loader:
        
        # Train model
        optimizer.zero_grad()

        pred = model(X_train_batch)
        loss = criterion(pred, y_train_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
    loss_list.append(loss)

    if iteration % 20 == 0:
        print(f"MSE: {loss}")
        
model.eval()
X_test = torch.tensor(X_test, dtype=torch.float)
preds = model(X_test)

print(f"""--- Random Forest ---
Mean Absolute Error: {mean_absolute_error(y_test, preds.detach().numpy())}
Mean Squared Error: {mean_squared_error(y_test, preds.detach().numpy())}
R2 Score: {r2_score(y_test, preds.detach().numpy())}""")

plt.scatter(preds.detach().numpy(), y_test, s=3)
plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], 'r-')
plt.savefig("nn.png")





# data = EmulatorData(directory=data_directory)
# data = data.drop_missing().drop_columns(columns=['exp_id', 'groupname', 'experiment'])
# data = data.create_boolean_indices()

# for col in data.data.columns:
#     data.data[col] = data.data[col].astype(float)
    
# data = data.split_data(target_column='ivol')
# data.X = data.scale(values=data.X, values_type='inputs', scaler="MinMaxScaler")
# data.y = data.scale(values=data.y, values_type='outputs', scaler="MinMaxScaler")
# data = data.train_test_split(shuffle=True)

# # rf = RandomForestRegressor(verbose=2, n_jobs=-1)
# # rf.fit(data.X_train, data.y_train)
# # preds = rf.predict(data.X_test)

# # print(f"""--- Random Forest ---
# # Mean Absolute Error: {mean_absolute_error(data.y_test, preds)}
# # Mean Squared Error: {mean_squared_error(data.y_test, preds)}
# # R2 Score: {r2_score(data.y_test, preds)}""")

# # plt.plot(data.y_test, preds, 'o')
# # plt.plot([0,1],[0,1],'-')
# # plt.savefig('rf.png')

# class Net(torch.nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.input_layer_size = data.X_train.shape[1]
#        self.layer = torch.nn.Linear(self.input_layer_size, 128)
#        self.layer2 = torch.nn.Linear(128, 32)
#        self.layer3 = torch.nn.Linear(32, 1)
#        self.relu = nn.ReLU()

#    def forward(self, x):
#        x = self.relu(self.layer(x))
#        x = self.relu(self.layer2(x))
#        x = self.layer3(x)     
#        return x
   
# class EmulatorDataset(Dataset):
    
#     def __init__(self, X_data, y_data):
#         self.X_data = X_data
#         self.y_data = y_data
        
#     def __getitem__(self, index):
#         return self.X_data[index], self.y_data[index]
        
#     def __len__ (self):
#         return len(self.X_data)

# X_train = np.array(data.X_train, dtype=np.float64)
# y_train = np.array(data.y_train, dtype=np.float64)
# X_test = np.array(data.X_test, dtype=np.float64)
# y_test = np.array(data.y_test, dtype=np.float64)
# train_dataset = EmulatorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
# test_dataset = EmulatorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

# train_loader = DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1)

# model = Net()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(),)
# epochs = 100
# verbose = True
# for epoch in range(epochs):
#     if verbose and epoch % 1 == 0:
#         print(f'---------- EPOCH: {epoch} ----------')

#     iteration = 0
#     for X_train_batch, y_train_batch in train_loader:
        
#         # Train model
#         optimizer.zero_grad()

#         pred = model(X_train_batch)
#         loss = criterion(pred, y_train_batch.unsqueeze(1))
#         loss.backward()
#         optimizer.step()

#     if iteration % 20 == 0:
#         print(f"MSE: {loss}")
        
# y_pred_list = []
# with torch.no_grad():
#     model.eval()
#     for X_batch, _ in test_loader:
#         y_test_pred = model(X_batch)
#         y_pred_list.append(y_test_pred.numpy())
# y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# # preds = preds.numpy().squeeze()
    
# print(f"""--- Random Forest ---
# Mean Absolute Error: {mean_absolute_error(y_test, y_pred_list)}
# Mean Squared Error: {mean_squared_error(y_test, y_pred_list)}
# R2 Score: {r2_score(y_test, y_pred_list)}""")

# plt.plot(y_test, y_pred_list, 'o')
# plt.plot([0,1],[0,1],'-')
# plt.savefig('nn.png')

stop = ''