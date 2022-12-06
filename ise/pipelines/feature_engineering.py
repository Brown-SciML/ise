from ise.data.EmulatorData import EmulatorData
from ise.utils.utils import _structure_emulatordata_args
import pandas as pd


def feature_engineer(data_directory, time_series, export_directory=None, emulator_data_args=None):
   
   
    emulator_data_args = _structure_emulatordata_args(input_args=emulator_data_args, time_series=time_series)
    
    emulator_data = EmulatorData(directory=data_directory)
    emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
        **emulator_data_args,
    )
    
    if export_directory:
        export_flag = 'ts' if time_series else 'traditional'
        train_features.to_csv(f'{export_directory}/{export_flag}_train_features.csv', index=False)
        test_features.to_csv(f'{export_directory}/{export_flag}_test_features.csv', index=False)
        pd.Series(train_labels, name='sle').to_csv(f'{export_directory}/{export_flag}_train_labels.csv', index=False)
        pd.Series(test_labels, name='sle').to_csv(f'{export_directory}/{export_flag}_test_labels.csv', index=False)
        pd.DataFrame(emulator_data.test_scenarios).to_csv(f'{export_directory}/{export_flag}_test_scenarios.csv', index=False)
        
    
    return emulator_data, train_features, test_features, train_labels, test_labels