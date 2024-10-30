from ise.models.predictors.deep_ensemble import DeepEnsemble
from ise.models.predictors.lstm import LSTM
from torch import nn, optim


class ISEFlow_AIS_DE(DeepEnsemble):
    def __init__(self, ):
        
        self.input_size = 99
        self.output_size = 1
        iseflow_ais_ensemble = [
            LSTM(1, 128, 99, 1, optim.HuberLoss()),
            LSTM(1, 512, 99, 1, optim.HuberLoss()),
            LSTM(1, 512, 99, 1, optim.HuberLoss()),
            LSTM(2, 128, 99, 1, optim.HuberLoss()),
            LSTM(1, 256, 99, 1, optim.L1Loss()),
            LSTM(1, 512, 99, 1, optim.MSELoss()),
            LSTM(2, 128, 99, 1, optim.MSELoss()),
            LSTM(2, 512, 99, 1, optim.MSELoss()),
            LSTM(1, 256, 99, 1, optim.L1Loss()),
            LSTM(1, 64, 99, 1, optim.HuberLoss()),
        ]
        super().__init__(self, ensemble_members=iseflow_ais_ensemble, input_size=self.input_size, output_size=self.output_size, output_sequence_length=86,)
    
    

class ISEFlow_GrIS_DE(DeepEnsemble):
    def __init__(self,):
        self.input_size = 91
        self.output_size = 1
        iseflow_gris_ensemble = [
            LSTM(2, 128, 99, 1, optim.HuberLoss()),
            LSTM(2, 256, 99, 1, optim.MSELoss()),
            LSTM(2, 128, 99, 1, optim.HuberLoss()),
            LSTM(2, 128, 99, 1, optim.MSELoss()),
            LSTM(2, 256, 99, 1, optim.HuberLoss()),
            LSTM(1, 256, 99, 1, optim.L1Loss()),
            LSTM(1, 128, 99, 1, optim.HuberLoss()),
            LSTM(2, 64, 99, 1, optim.MSELoss()),
            LSTM(2, 256, 99, 1, optim.HuberLoss()),
            LSTM(1, 256, 99, 1, optim.L1Loss()),
        ]
        super().__init__(self, ensemble_members=iseflow_gris_ensemble, input_size=self.input_size, output_size=self.output_size, output_sequence_length=86,)