import sys

sys.path.append("../..")

import numpy as np
from ise.data.feature_engineer import FeatureEngineer
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import r2_score

from ise.models.grid import WeakPredictor
from ise.models.loss import WeightedMSELoss, WeightedMSELossWithSignPenalty, WeightedMSEPCALoss, WeightedPCALoss
from ise.evaluation.metrics import mape, relative_squared_error
from ise.utils.functions import get_X_y

ice_sheet = 'AIS'
dataset = 'sectors'
loss = 'MSELoss'
lag = 5
overwrite_data = True
scale = True

train = True
epochs = 1


dir_ = f"/oscar/scratch/pvankatw/datasets/{dataset}/{ice_sheet}/"
experiment_description = f'{dataset}_scaled{scale}_lag{lag}_{loss}'
print('Experiment Description:', experiment_description)
print('Ice Sheet:', ice_sheet)

if not train and not os.path.exists(f"{dir_}/WeakPredictorModel_{experiment_description}.pth"):
    raise FileNotFoundError(f"{dir_}/WeakPredictorModel_{experiment_description}.pth does not exist. Please train the model first.")

grids_directory = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Grid_Files/"
grids_file = f"{grids_directory}/AIS_sectors_8km.nc" if ice_sheet=='AIS' else f"{grids_directory}/GrIS_Basins_Rignot_sectors_5km.nc"


train_exists = os.path.exists(f"{dir_}/train.csv")
if not train_exists or (train_exists and overwrite_data):
    fe = FeatureEngineer(ice_sheet=ice_sheet, data=pd.read_csv(f"{dir_}/dataset.csv"))
    fe.fill_mrro_nans(method='mean')
    fe.add_lag_variables(lag=lag)
    quantile = 0.005
    fe.drop_outliers(method='explicit', column='sle', expression=[('sle', '<', np.percentile(fe.data.sle, quantile*100))])
    if scale:
        fe.scale_data(save_dir=f"{dir_}/")
    fe.split_data(train_size=0.7, val_size=0.15, test_size=0.15, output_directory=dir_)
    data = fe.train
else:
    data = pd.read_csv(f"{dir_}/train.csv")
    
# data = pd.read_csv('/users/pvankatw/research/A_variational_LSTM_emulator/emulator/untracked_folder/ml_data/ts_train_features.csv')
# data = data[data.columns[0:55]]
# data = data.sort_values(by=['model', 'exp', 'sector', 'year'])
X, y = get_X_y(pd.read_csv(f"{dir_}/train.csv"), 'sectors', return_format='numpy')
val_X, val_y = get_X_y(pd.read_csv(f"{dir_}/val.csv"), 'sectors', return_format='numpy')


# X = data
# y = pd.read_csv('/users/pvankatw/research/A_variational_LSTM_emulator/emulator/untracked_folder/ml_data/ts_train_labels.csv')


# dim_processor = DimensionProcessor(
#     pca_model=f"{dir_}/pca_models/{ice_sheet}_pca_sle.pth", 
#     scaler_model=f"{dir_}/scalers/{ice_sheet}_scaler_sle.pth"
# )
dim_processor = None


losses = dict(WeightedMSELoss=WeightedMSELoss(y.mean().mean(), y.flatten().std(),), 
            # WeightedMSEPCALoss=WeightedMSEPCALoss(y.mean().mean(), y.values.flatten().std(), component_weights),
            # WeightedPCALoss=WeightedPCALoss(component_weights),
            HuberLoss=torch.nn.HuberLoss(),
            MSELoss=torch.nn.MSELoss(),
            WeightedMSELossWithSignPenalty=WeightedMSELossWithSignPenalty(y.mean().mean(), y.flatten().std(), weight_factor=1.0, sign_penalty_factor=0.5),
            )

model = WeakPredictor(
    input_size=X.shape[1], 
    lstm_num_layers=1, 
    lstm_hidden_size=512, 
    output_size=1,
    dim_processor=dim_processor,
    ice_sheet = ice_sheet,
    )
if train:
    model.fit(X, y, epochs=epochs, sequence_length=5, batch_size=256, loss=losses[loss], val_X=val_X, val_y=val_y)
    torch.save(model.state_dict(), f"{dir_}/WeakPredictorModel_{experiment_description}.pth")
else:
    model.load_state_dict(torch.load(f"{dir_}/WeakPredictorModel_{experiment_description}.pth", map_location=torch.device('cpu')), )

model.eval()
X = val_X
y = val_y

y_preds = model.predict(X).cpu().detach().numpy()
comparison = pd.DataFrame(dict(y=y, y_preds=y_preds.flatten()))
comparison['diff'] = (comparison.y_preds - comparison.y).values
comparison['se'] = (comparison.y_preds - comparison.y).values **2

if scale:
    fe = FeatureEngineer(ice_sheet=ice_sheet, data=pd.read_csv(f"{dir_}/dataset.csv"))
    _, y = fe.unscale_data(y=y, scaler_y_path=f"{dir_}/scaler_y.pkl")
    _, y_pred = fe.unscale_data(y=y_preds, scaler_y_path=f"{dir_}/scaler_y.pkl")
    
# mse = pd.DataFrame(dict(mse=((y_preds-y)**2).sle, y_true=y.sle.squeeze(), y_pred=y_preds.flatten()))
mse = np.mean((y_pred.flatten() - y) ** 2)
mae = np.mean(abs((y_pred.flatten() - y.sle.values)))
rmse = np.sqrt(mse)
r2 = r2_score(y.sle.values, y_pred.flatten())

print(f"""Validation MSE: {mse}
-- MAE: {mae}
-- RMSE: {rmse}
-- R2: {r2}
-- Relative Squared Error: {relative_squared_error(y.sle.values, y_pred.flatten())}
-- Mean Absolute Percentage Error: {mape(y.sle.values, y_pred.flatten())}""")

scenarios = data[['year', 'sector', 'aogcm', 'exp', 'model', 'Scenario', ]]
scenarios['predicted'], scenarios['true'] = y_preds, y
scenarios['squared_error'] = (scenarios['predicted'] - scenarios['true'])**2
sector_errors = scenarios.groupby('sector').mean()['squared_error']
scenario_timeseries = scenarios.groupby(['Scenario', 'year']).mean()['predicted']

plt.plot(np.arange(2015,2101), scenario_timeseries['rcp2.6'], label='RCP2.6')
plt.plot(np.arange(2015,2101), scenario_timeseries['rcp8.5'], label='RCP8.5')
plt.title('Average Projection by Scenario')
plt.legend()
plt.savefig(f'./supplemental/{ice_sheet}scenarios.png')
plt.close('all')


for i in [0, 5, 10, 15, 20, 25]:
    plt.plot(y_pred[i*86:i*86+86, :], label='Predicted')
    plt.plot(y.values[i*86:i*86+86, :], label='True')
    plt.title('True v Predicted')
    plt.legend()
    plt.savefig(f'/users/pvankatw/research/current/supplemental/sectors/example_plots/{i}.png')
    plt.close('all')
    
stop = ''