from ise.models.predictors.deep_ensemble import DeepEnsemble
from ise.models.predictors.lstm import LSTM
from torch import nn, optim

AIS_input_size = 99
AIS_output_size = 1
iseflow_ais_ensemble = [
    LSTM(1, 128, 99, 1, optim.HuberLoss(),),
    
    
]
class ISEFlow_AIS_DE(DeepEnsemble):
    def __init__()