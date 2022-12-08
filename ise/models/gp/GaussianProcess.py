

import pandas as pd
import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import random

# TODO: Make this a model class

train_features = pd.read_csv(r"/users/pvankatw/emulator/src/data/ml/dataset4/test_features.csv")
train_labels = pd.read_csv(r"/users/pvankatw/emulator/src/data/ml/dataset4/train_labels.csv")
test_features = pd.read_csv(r"/users/pvankatw/emulator/src/data/ml/dataset4/test_features.csv")
test_labels = pd.read_csv(r"/users/pvankatw/emulator/src/data/ml/dataset4/test_labels.csv")


kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=9
)
# gaussian_process.fit(train_features, train_labels)

# MemoryError: Unable to allocate 236. GiB for an array with shape (177898, 177898) and data type float64


# mean_prediction, std_prediction = gaussian_process.predict(test_features, return_std=True)



n = 25000
print(f'training with {n} samples')

tf_samples = train_features.sample(n)
# gp_training_data = tf_samples[['temperature', 'salinity']]
gp_train_features = tf_samples['temperature']
gp_train_labels = np.array(train_labels.loc[gp_train_features.index]).reshape(-1, 1)

if isinstance(gp_train_features, pd.Series) or gp_train_features.shape[1] == 1:
    gp_train_features = np.array(gp_train_features).reshape(-1, 1)



kernel = RBF()
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=9
)
gaussian_process.fit(gp_train_features, gp_train_labels,)



gp_test_features = test_features['temperature']
gp_test_labels = np.array(test_labels).reshape(-1, 1)

if isinstance(gp_test_features, pd.Series) or gp_test_features.shape[1] == 1:
    gp_test_features = np.array(gp_test_features).reshape(-1, 1)
    
    
mean_prediction, std_prediction = gaussian_process.predict(gp_test_features, return_std=True)
preds = mean_prediction

print(f"""--- Random Forest ---
Mean Absolute Error: {mean_absolute_error(test_labels, preds)}
Mean Squared Error: {mean_squared_error(test_labels, preds)}
R2 Score: {r2_score(test_labels, preds)}""")

pd.Series(dict(preds=preds,)).to_csv(f'gp_preds_n{n}_temperature.csv')
pd.Series(dict(std=std_prediction)).to_csv(f'gp_std_n{n}_temperature.csv')

plt.plot(preds, test_labels, 'o')
plt.plot([min(gp_test_labels), max(gp_test_labels)], [min(gp_test_labels), max(gp_test_labels)], '-')
plt.savefig(f'GP_results.png')



test_scenarios = pd.read_csv(r"/users/pvankatw/emulator/src/data/ml/dataset4/test_scenarios.csv").values.tolist()


sectors = list(set(test_features.sectors))
sectors.sort()
draws = 'random'
k = 5
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
    single_test_features = np.array(test_features[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)]['temperature'], dtype=np.float64)
    single_test_labels = np.array(test_labels[(test_features[test_model] == 1) & (test_features[test_exp] == 1) & (test_features.sectors == test_sector)], dtype=np.float64)
    preds, std = gaussian_process.predict(single_test_features.reshape(-1,1), return_std=True)

#     plt.figure(figsize=(15,8))
    plt.figure()
    plt.plot(single_test_labels, 'r-', label='True')
    plt.plot(preds, 'b-', label='Predicted')
    plt.xlabel('Time (years since 2015)')
    plt.ylabel('SLE (mm)')
    plt.title(f'Model={test_model}, Exp={test_exp}, sector={sectors.index(test_sector)+1}')
    
    plt.fill_between(
    np.arange(0,85).ravel(),
    preds - 1.96 * std,
    preds + 1.96 * std,
    alpha=0.5,
    label=r"95% confidence interval",
)
        
    plt.legend()
    plt.savefig(f'GP_{test_model}_{test_exp}_test_sector.png')

    