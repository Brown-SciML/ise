import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time

data = pd.read_csv(r"/users/pvankatw/emulator/src/data/output_files/master.csv", low_memory=False)
data = data[['salinity', 'temperature', 'thermal_forcing', 'pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly',
       'smb_anomaly', 'ts_anomaly', 'ivaf']]

data = data.dropna()
dataset = pd.get_dummies(data)

for col in dataset.columns:
    dataset[col] = dataset[col].astype(float)
    
prescaled_data = pd.get_dummies(dataset)
scaler = MinMaxScaler()
scaler.fit(prescaled_data)
dataset = pd.DataFrame(scaler.transform(prescaled_data), columns=prescaled_data.columns)

train_dataset = dataset.sample(frac=0.1, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.drop(columns=['ivaf'])
test_features = test_dataset.drop(columns=['ivaf'])
train_labels = train_dataset['ivaf']
test_labels = test_dataset['ivaf']

start = time.time()
kernel = RBF()
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=9
)
print('Training')
gaussian_process.fit(train_features, train_labels,)
end_training = time.time()

print('Predicting')
mean_prediction, std_prediction = gaussian_process.predict(test_features, return_std=True)
end_prediction = time.time()

preds = mean_prediction

print(f"""--- Gaussian Process ---
Mean Absolute Error: {mean_absolute_error(test_labels, preds)}
Mean Squared Error: {mean_squared_error(test_labels, preds)}
R2 Score: {r2_score(test_labels, preds)}""")

plt.plot(preds, test_labels, 'o')
plt.plot([0,1], [0,1], '-')
plt.savefig('gp.png')

print('')
print(f"""Total Time: {(end_prediction - end_training)} seconds
   - training: {(end_training - start)} seconds
   - prediction: {(end_prediction - end_training)} seconds
   
Total training rows: {len(train_features)}
   - Rows/second: {len(train_features) // (end_training - start)}""")