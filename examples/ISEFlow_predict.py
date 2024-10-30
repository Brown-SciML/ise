from ise.models.ISEFlow import ISEFlow_AIS, ISEFlow_GrIS
from ise.utils import get_data, unscale
import numpy as np

iseflowais = ISEFlow_AIS.load(version="v1.0.0")

data_directory = r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/AIS/"
X_train, y_train, X_val, y_val, X_test, y_test = get_data(data_directory, return_format='numpy')
y_test = unscale(y_test.reshape(-1,1), f"{data_directory}/scaler_y.pkl")

preds, uq = iseflowais.predict(X_test, )

print("MSE: ", np.mean((y_test - preds)**2)) # reported in ISEFlow paper: 1.20

