from ise.models.ISEFlow import ISEFlow
from ise.models.density_estimators.normalizing_flow import NormalizingFlow
from ise.models.predictors.lstm import LSTM
from ise.utils.functions import to_tensor
from ise.utils import functions as f
from ise.models.predictors.deep_ensemble import DeepEnsemble
import pandas as pd
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


flow = NormalizingFlow.load(r"/users/pvankatw/research/ise/model/GrIS/nf/nf.pt")
flow.trained = True

ensemble_members = ['lstm_22126', 'lstm_17567', 'lstm_24104', 'lstm_3993', 'lstm_30488', 'lstm_17239', 'lstm_30799', 'lstm_21682', 'lstm_468', 'lstm_5086']
# ensemble_members = ['lstm_6827', 'lstm_14993', 'lstm_26595', 'lstm_4500', 'lstm_4271', 'lstm_8468', 'lstm_13654', 'lstm_31659', 'lstm_4084', 'lstm_17368']
ensemble_models = []
for member in ensemble_members:
    model = LSTM.load(f"/users/pvankatw/research/ise/model/GrIS/lstm/{member}/lstm.pt")
    ensemble_models.append(model)

ensemble = DeepEnsemble(
    ensemble_models,
    76,
    num_ensemble_members=len(ensemble_models),
)
# ensemble.sequence_length = 10

iseflow = ISEFlow(
    ensemble, flow,
)

iseflow.save(r"/users/pvankatw/research/ise/model/GrIS/ISEFlow_slc", output_scaler_path=r"/users/pvankatw/research/ise/dataset/GrIS_slc/scalers/scaler_y.pkl")

