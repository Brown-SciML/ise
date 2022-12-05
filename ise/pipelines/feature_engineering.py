from ise.data.EmulatorData import EmulatorData
import pandas as pd


def feature_engineer(data_dir, time_series, export_directory=None, target_column='sle', drop_missing=True, drop_columns=['groupname', 'experiment'], boolean_indices=True, scale=True, drop_outliers={'column': 'ivaf', 'operator': '<', 'value': -1e13}, lag=5):
   
    emulator_data = EmulatorData(directory=data_dir)
    emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
        target_column=target_column,
        drop_missing=drop_missing,
        drop_columns=drop_columns,
        boolean_indices=boolean_indices,
        scale=scale,
        split_type='batch_test',
        drop_outliers=drop_outliers,
        time_series=time_series,
        lag=lag,  # TODO: update with results from lag_sequence_test
    )
    
    if export_directory:
        export_flag = 'ts' if time_series else 'traditional'
        train_features.to_csv(f'{export_directory}/{export_flag}_train_features.csv', index=False)
        test_features.to_csv(f'{export_directory}/{export_flag}_test_features.csv', index=False)
        pd.Series(train_labels, name='sle').to_csv(f'{export_directory}/{export_flag}_train_labels.csv', index=False)
        pd.Series(test_labels, name='sle').to_csv(f'{export_directory}/{export_flag}_test_labels.csv', index=False)
        pd.DataFrame(emulator_data.test_scenarios).to_csv(f'{export_directory}/{export_flag}_test_scenarios.csv', index=False)
        
    
    return emulator_data, train_features, test_features, train_labels, test_labels