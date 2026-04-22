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


flow = NormalizingFlow.load(r"/users/pvankatw/research/ise/model/AIS/nf/nf.pt")
flow.trained = True

# ensemble_members = ['lstm_6827', 'lstm_14993', 'lstm_26595', 'lstm_4084', 'lstm_4271', 'lstm_31659', 'lstm_17368', 'lstm_23149']
# ensemble_members = ['lstm_6827', 'lstm_14993', 'lstm_26595', 'lstm_4500', 'lstm_4271', 'lstm_8468', 'lstm_13654', 'lstm_31659', 'lstm_4084', 'lstm_17368']
ensemble_members = ['lstm_19329', 'lstm_24965', 'lstm_8394', 'lstm_28054', 'lstm_30516', 'lstm_24942', 'lstm_15804', 'lstm_5346', 'lstm_2873', 'lstm_29567']
ensemble_members = ['lstm_13250', 'lstm_22022', 'lstm_19577', 'lstm_3236', 'lstm_9495', 'lstm_16592', 'lstm_20692', 'lstm_1004', 'lstm_22488', 'lstm_7879']
ensemble_members = ['lstm_5406', "lstm_6116", "lstm_13055", "lstm_18722", "lstm_26619", "lstm_16298", "lstm_3865", "lstm_15381", "lstm_26999", "lstm_17129"]
ensemble_models = []
for member in ensemble_members:
    model = LSTM.load(f"/users/pvankatw/research/ise/model/AIS/lstm/{member}/lstm.pt")
    ensemble_models.append(model)

ensemble = DeepEnsemble(
    ensemble_models,
    94,
    num_ensemble_members=len(ensemble_models),
)
ensemble.sequence_length = 10

iseflow = ISEFlow(
    ensemble, flow,
)

iseflow.save(r"/users/pvankatw/research/ise/model/AIS/ISEFlow_slc_full/", output_scaler_path=r"/users/pvankatw/research/ise/dataset/AIS_slc_all/scalers/scaler_y.pkl")

