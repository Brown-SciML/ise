"""Pipeline for feature engineering. After data has been processed from the raw NC files using 
ise.pipelines.processing, this module will get data ready for modeling."""
from ise.data.EmulatorData import EmulatorData
from ise.utils.utils import _structure_emulatordata_args
import pandas as pd


def feature_engineer(data_directory: str, time_series: bool=True, export_directory: str=None, emulator_data_args: dict=None):
    """Performs feature engineering after ise.pipelines.processing has been run. Includes loading
    data, formatting, processing, and splitting data to get data into training and testing sets.

    Args:
        data_directory (str): Directory containing training and testing data.
        time_series (bool): Flag denoting wether model was trained with time-series data. Defaults to True.
        export_directory (str, optional): Directory to save exported files. Defaults to None.
        emulator_data_args (dict, optional): Kwarg arguments to EmulatorData.process. Default will keep optimal values, see EmulatorData.process for more details. Defaults to None.

    Returns:
        tuple: Tuple containing [emulator_data, train_features, test_features, train_labels, test_labels], or the EmulatorData class and the associate training and testing datasets.
    """
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