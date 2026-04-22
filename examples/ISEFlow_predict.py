from ise.models.ISEFlow import ISEFlow_AIS, ISEFlow_GrIS, ISEFlow
from ise.models.density_estimators.normalizing_flow import NormalizingFlow
from ise.models.predictors.deep_ensemble import DeepEnsemble
from ise.models.pretrained import ISEFlow_AIS_v1_0_0_path, ISEFlow_GrIS_v1_0_0_path
from ise.utils import get_data, unscale
import numpy as np

ice_sheet = "AIS"

model_paths = ISEFlow_GrIS_v1_0_0_path if ice_sheet == "GrIS" else ISEFlow_AIS_v1_0_0_path

data_directory = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ice_sheet}/"
X_train, y_train, X_val, y_val, X_test, y_test = get_data(data_directory, return_format='numpy')
y_test = unscale(y_test.reshape(-1,1), f"{data_directory}/scaler_y.pkl")

de = DeepEnsemble.load(f"{model_paths}/deep_ensemble.pth")
nf = NormalizingFlow.load(f"{model_paths}/normalizing_flow.pth")
iseflowais = ISEFlow(de, nf)
preds, uq = iseflowais.predict(X_test, )

# iseflowais = ISEFlow.load(version="v1.0.0")

data_directory = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ice_sheet}/"
X_train, y_train, X_val, y_val, X_test, _ = get_data(data_directory, return_format='pandas')

X_test['aleatoric'] = uq['aleatoric']
X_test['epistemic'] = uq['epistemic']
X_test['preds'] = preds
X_test['true'] = y_test
X_test.to_csv(f"iseflow_preds_{ice_sheet}.csv")
print("MSE: ", np.mean((y_test - preds)**2)) # reported in ISEFlow paper: 1.20

