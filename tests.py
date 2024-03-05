
from ise.grids.models.HybridEmulator import WeakPredictor, DimensionProcessor
from ise.grids.data.feature_engineer import FeatureEngineer
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from ise.grids.visualization.evaluation import EvaluationPlotter
from ise.grids.models.loss import WeightedMSELoss, WeightedMSEPCALoss, WeightedPCALoss


ice_sheet = 'AIS'
print('Ice Sheet:', ice_sheet)
dir_ = f"/oscar/scratch/pvankatw/pca/{ice_sheet}"

for lag in [None, 3, 5]:
    print('________________________________________________________________________')
    print('Lag:', lag)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fe = FeatureEngineer(ice_sheet=ice_sheet, data=pd.read_csv(f"{dir_}/dataset.csv"))
    fe.fill_mrro_nans(method='drop')
    if lag is not None:
        fe.add_lag_variables(lag=lag)
    # fe.backfill_outliers(percentile=99.999)
    fe.split_data(train_size=0.7, val_size=0.15, test_size=0.15, output_directory=f"/oscar/scratch/pvankatw/pca/{ice_sheet}/")

    data = fe.train

    X = data.drop(columns=[x for x in data.columns if 'sle' in x] + ['cmip_model', 'pathway', 'exp', 'id'])
    y = data[[x for x in data.columns if 'sle' in x]]

    component_weights = np.ones(len(y.columns))
    component_weights[0:10] = [100, 50, 30, 20, 10, 10, 10, 5, 5, 5]
    component_weights[-50:] = 0.1*np.ones(50)

    losses = dict(WeightedMSELoss=WeightedMSELoss(y.mean().mean(), y.values.flatten().std(),).to(device), 
                WeightedMSEPCALoss=WeightedMSEPCALoss(y.mean().mean(), y.values.flatten().std(), component_weights).to(device),
                WeightedPCALoss=WeightedPCALoss(component_weights).to(device),
                HuberLoss=torch.nn.HuberLoss(),
                MSELoss=torch.nn.MSELoss(),
                )


    dim_processor = DimensionProcessor(
        pca_model=f"/oscar/scratch/pvankatw/pca/{ice_sheet}/pca_models/{ice_sheet}_pca_sle.pth", 
        scaler_model=f"/oscar/scratch/pvankatw/pca/{ice_sheet}/scalers/{ice_sheet}_scaler_sle.pth"
    )

    model = WeakPredictor(
        input_size=len(X.columns), 
        lstm_num_layers=3, 
        lstm_hidden_size=512, 
        output_size=len(y.columns),
        dim_processor=dim_processor,
        ice_sheet = ice_sheet,
        )
    
    for loss_name, loss in losses.items():
        experiment_description = f'lag{lag}_{loss_name}'
        
        model.fit(X, y, epochs=100, sequence_length=3, batch_size=128, loss=loss)
        torch.save(model.state_dict(), f"{dir_}/WeakPredictorModel_lag{lag}_{loss_name}.pth")
        # model.load_state_dict(torch.load(f"{dir_}/WeakPredictorModel_lag{lag}_{loss_name}.pth", map_location=torch.device('cpu')), )

        model.eval()
        data = fe.val


        for i in [10, 20, 35]:
            X = data.drop(columns=[x for x in data.columns if 'sle' in x] + ['cmip_model', 'pathway', 'exp', 'id'])
            # X = data.drop(columns=[x for x in data.columns if 'sle' in x] + ['id'])
            y = data[[x for x in data.columns if 'sle' in x]]
            i = 20
            x = X.iloc[i*86:i*86+86, :].values
            y = y.iloc[i*86:i*86+86, :].values
            y_pred = model.predict(x)
            y_pred = y_pred.cpu().detach().numpy()

            y_pred = dim_processor.to_grid(y_pred, unscale=True).squeeze().reshape(86, 761, 761)
            y_true = dim_processor.to_grid(y, unscale=True).squeeze().reshape(86, 761, 761)

            plotter = EvaluationPlotter(save_dir='/users/pvankatw/research/current/supplemental/training_experiments')
            plotter.spatial_side_by_side(y_true.cpu(), y_pred.cpu(), timestep=86, save_path=f'{experiment_description}_{i}.png', cmap=plt.cm.viridis, video=False)

    stop = ''