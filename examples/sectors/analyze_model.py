import pandas as pd
import sys
sys.path.append("../..")
from ise.models.timeseries import TimeSeriesEmulator
from ise.pipelines.testing import analyze_model


DATA_DIRECTORY = r"/users/pvankatw/emulator/untracked_folder/ml_data_directory"
PRETRAINED_MODELS = r'/users/pvankatw/emulator/ise/models/pretrained/'
UNTRACKED = r'/users/pvankatw/emulator/untracked_folder'


train_features = pd.read_csv(f"{DATA_DIRECTORY}/ts_train_features.csv")
architecture = {
    'num_rnn_layers': 4,
    'num_rnn_hidden': 256,
    'input_layer_size': train_features.shape[1]
}

model_path = f'{PRETRAINED_MODELS}/Emulator.pt'

print('\nAnalyzing')
analyze_model(
    data_directory=DATA_DIRECTORY,
    model_path=model_path,
    architecture=architecture,
    model_class=TimeSeriesEmulator,
    time_series=True,
    mc_dropout=True,
    dropout_prob=0.3,
    mc_iterations=100,
    verbose=False,
    save_directory=f'{UNTRACKED}/analyze_model'
)