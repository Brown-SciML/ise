from ise.models.predictors.deep_ensemble import DeepEnsemble
from ise.models.predictors.lstm import LSTM
from torch import nn, optim


class ISEFlow_AIS_DE(DeepEnsemble):
    """
    ISEFlow Deep ensemble model for Antarctic Ice Sheet (AIS) emulation.

    This class implements an ensemble of Long Short-Term Memory (LSTM) networks 
    to predict ice sheet dynamics using deep learning. It extends the `DeepEnsemble` 
    class and combines multiple LSTM models to enhance predictive performance.

    Attributes:
        input_size (int): The number of input features, Defaults to 99.
        output_size (int): The number of output features, Defaults to 1.
        iseflow_ais_ensemble (list): A list of LSTM models with different architectures and loss functions.
    
    Inherits from:
        DeepEnsemble: A base class for deep ensemble models.

    """

    def __init__(self, ):
        
        self.input_size = 99
        self.output_size = 1
        iseflow_ais_ensemble = [
            LSTM(1, 128, 99, 1, nn.HuberLoss()),
            LSTM(1, 512, 99, 1, nn.HuberLoss()),
            LSTM(1, 512, 99, 1, nn.HuberLoss()),
            LSTM(2, 128, 99, 1, nn.HuberLoss()),
            LSTM(1, 256, 99, 1, nn.L1Loss()),
            LSTM(1, 512, 99, 1, nn.MSELoss()),
            LSTM(2, 128, 99, 1, nn.MSELoss()),
            LSTM(2, 512, 99, 1, nn.MSELoss()),
            LSTM(1, 256, 99, 1, nn.L1Loss()),
            LSTM(1, 64, 99, 1, nn.HuberLoss()),
        ]
        super().__init__(ensemble_members=iseflow_ais_ensemble, input_size=self.input_size, output_size=self.output_size, output_sequence_length=86,)
    
    

class ISEFlow_GrIS_DE(DeepEnsemble):
    """
    ISEFlow Deep ensemble model for Greenland Ice Sheet (GrIS) emulation.

    This class constructs an ensemble of LSTM models to predict ice sheet behavior 
    for the Greenland Ice Sheet (GrIS). It extends the `DeepEnsemble` framework 
    and integrates multiple LSTM-based predictors to improve accuracy.

    Attributes:
        input_size (int): The number of input features (90).
        output_size (int): The number of output features (1).
        iseflow_gris_ensemble (list): A list of LSTM models with varying architectures and loss functions.

    Inherits from:
        DeepEnsemble: A base class for deep ensemble models.
    """

    def __init__(self,):
        self.input_size = 90
        self.output_size = 1
        iseflow_gris_ensemble = [
            LSTM(2, 128, 99, 1, nn.HuberLoss()),
            LSTM(2, 256, 99, 1, nn.MSELoss()),
            LSTM(2, 128, 99, 1, nn.HuberLoss()),
            LSTM(2, 128, 99, 1, nn.MSELoss()),
            LSTM(2, 256, 99, 1, nn.HuberLoss()),
            LSTM(1, 256, 99, 1, nn.L1Loss()),
            LSTM(1, 128, 99, 1, nn.HuberLoss()),
            LSTM(2, 64, 99, 1, nn.MSELoss()),
            LSTM(2, 256, 99, 1, nn.HuberLoss()),
            LSTM(1, 256, 99, 1, nn.L1Loss()),
        ]
        super().__init__(ensemble_members=iseflow_gris_ensemble, input_size=self.input_size, output_size=self.output_size, output_sequence_length=86,)