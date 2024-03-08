from ise.grids.models.HybridEmulator import WeakPredictor, DimensionProcessor
from ise.grids.data.feature_engineer import FeatureEngineer
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from ise.grids.visualization.evaluation import EvaluationPlotter
from ise.grids.models.loss import WeightedMSELoss, WeightedMSEPCALoss, WeightedPCALoss, WeightedMSELossWithSignPenalty
import os
from ise.grids.evaluation.metrics import sum_by_sector, mean_squared_error_sector


ice_sheet = 'GrIS'
dataset = 'sectors'
loss = 'MSELoss'
lag = 3
overwrite_data = False
train = False
scale_pcas = False


dir_ = f"/oscar/scratch/pvankatw/datasets/{dataset}/{ice_sheet}/"

experiment_description = f'{dataset}_scaled{scale_pcas}_lag{lag}_{loss}'
print('Experiment Description:', experiment_description)
print('Ice Sheet:', ice_sheet)

if not train and not os.path.exists(f"{dir_}/WeakPredictorModel_{experiment_description}.pth"):
    raise FileNotFoundError(f"{dir_}/WeakPredictorModel_{experiment_description}.pth does not exist. Please train the model first.")

grids_directory = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Grid_Files/"
grids_file = f"{grids_directory}/AIS_sectors_8km.nc" if ice_sheet=='AIS' else f"{grids_directory}/GrIS_Basins_Rignot_sectors_5km.nc"


train_exists = os.path.exists(f"{dir_}/train.csv")
if not train_exists or (train_exists and overwrite_data):
    fe = FeatureEngineer(ice_sheet=ice_sheet, data=pd.read_csv(f"{dir_}/dataset.csv"))
    fe.fill_mrro_nans(method='drop')
    fe.add_lag_variables(lag=3)
    if scale_pcas:
        fe.scale_data(save_dir=f"{dir_}/scalers/")
    fe.split_data(train_size=0.7, val_size=0.15, test_size=0.15, output_directory=dir_)
    data = fe.train
else:
    data = pd.read_csv(f"{dir_}/train.csv")
    
data.sort_values(by=['model', 'exp', 'sector', 'year'])

try:
    X_drop = [x for x in data.columns if 'sle' in x] + ['cmip_model', 'pathway', 'exp', 'id']
    X = data.drop(columns=X_drop)
except:
    X_drop = [x for x in data.columns if 'sle' in x] + ['id']
    X = data.drop(columns=X_drop)
y = data[[x for x in data.columns if 'sle' in x]]


dim_processor = DimensionProcessor(
    pca_model=f"{dir_}/pca_models/{ice_sheet}_pca_sle.pth", 
    scaler_model=f"{dir_}/scalers/{ice_sheet}_scaler_sle.pth"
)

if dataset == 'pca_full':
    component_weights = np.ones(len(y.columns))
    component_weights[0:10] = [100, 50, 30, 20, 10, 10, 10, 5, 5, 5]
    component_weights[-50:] = 0.1*np.ones(50)
elif dataset == 'pca_18':
    component_weights = np.ones(len(y.columns))
    component_weights[0:10] = [100, 50, 30, 20, 10, 10, 10, 5, 5, 5]

losses = dict(WeightedMSELoss=WeightedMSELoss(y.mean().mean(), y.values.flatten().std(),), 
            WeightedMSEPCALoss=WeightedMSEPCALoss(y.mean().mean(), y.values.flatten().std(), component_weights),
            WeightedPCALoss=WeightedPCALoss(component_weights),
            HuberLoss=torch.nn.HuberLoss(),
            MSELoss=torch.nn.MSELoss(),
            WeightedMSELossWithSignPenalty=WeightedMSELossWithSignPenalty(y.mean().mean(), y.values.flatten().std(), weight_factor=1.0, sign_penalty_factor=0.01),
            )

model = WeakPredictor(
    input_size=len(X.columns), 
    lstm_num_layers=3, 
    lstm_hidden_size=512, 
    output_size=len(y.columns),
    dim_processor=dim_processor,
    ice_sheet = ice_sheet,
    )
print(model.device)
if train:
    model.fit(X, y, epochs=100, sequence_length=3, batch_size=128, loss=losses[loss])
    torch.save(model.state_dict(), f"{dir_}/WeakPredictorModel_{experiment_description}.pth")
else:
    model.load_state_dict(torch.load(f"{dir_}/WeakPredictorModel_{experiment_description}.pth", map_location=torch.device('cpu')), )

model.eval()

path = f"{dir_}/val.csv"
data = pd.read_csv(path)

shape = (86, 761, 761) if ice_sheet=='AIS' else (86, 337, 577)

X = data.drop(columns=X_drop)
y = data[[x for x in data.columns if 'sle' in x]]

all_preds = dim_processor.to_grid(model.predict(X)).reshape(-1, 761, 761) if ice_sheet=='AIS' else dim_processor.to_grid(model.predict(X)).reshape(-1, 337, 577)
all_trues = dim_processor.to_grid(y).reshape(-1, 761, 761) if ice_sheet=='AIS' else dim_processor.to_grid(y).reshape(-1, 337, 577)
sector_mse = mean_squared_error_sector(sum_by_sector(all_trues, grids_file), sum_by_sector(all_preds, grids_file))
test_mse = torch.mean((all_preds - all_trues)**2)
print(f"Test MSE: {test_mse:0.6f}")
print(f"Sector MSE: {sector_mse:0.2f}")

for i in [0, 5, 10, 15, ]:
    X = data.drop(columns=X_drop)
    y = data[[x for x in data.columns if 'sle' in x]]
    x = X.iloc[i*86:i*86+86, :].values
    y = y.iloc[i*86:i*86+86, :].values
    y_pred = model.predict(x)
    y_pred = y_pred.cpu().detach().numpy()
    
    if scale_pcas:
        fe = FeatureEngineer(ice_sheet=ice_sheet, data=pd.read_csv(f"{dir_}/dataset.csv"))
        _, y = fe.unscale_data(y=y, scaler_y_path=f"{dir_}/scalers/scaler_y.pkl")
        _, y_pred = fe.unscale_data(y=y_pred, scaler_y_path=f"{dir_}/scalers/scaler_y.pkl")

    y_pred = dim_processor.to_grid(y_pred, unscale=True).squeeze().reshape(shape)
    y_true = dim_processor.to_grid(y, unscale=True).squeeze().reshape(shape)
    
    sum_sectors_pred = sum_by_sector(y_pred, grids_file)
    sum_sectors_true = sum_by_sector(y_true, grids_file)
    
    plotter = EvaluationPlotter(save_dir=f'/users/pvankatw/research/current/supplemental/more_training_experiments/{ice_sheet}/{dataset}/')
    plotter.sector_side_by_side(sum_sectors_true, sum_sectors_pred, grids_file, outline_array_true=y_true, outline_array_pred=y_pred, timestep=86, save_path=f'{experiment_description}_mse{test_mse:0.2f}_sectors_{i}.png', cmap=plt.cm.viridis,)
    plotter.spatial_side_by_side(y_true.cpu(), y_pred.cpu(), timestep=86, save_path=f'{experiment_description}_mse{test_mse:0.2f}_{i}.png', cmap=plt.cm.viridis, video=False)



stop = ''