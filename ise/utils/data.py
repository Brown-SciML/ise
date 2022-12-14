import pandas as pd
from ise.utils.utils import _structure_emulatordata_args
from ise.data import EmulatorData
from itertools import product
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon


def load_ml_data(data_directory, time_series):
    """Loads training and testing data for machine learning models. These files are generated using 
    functions in the ise.data.processing modules or process_data in the ise.pipelines.processing module.

    Args:
        data_directory (str): Directory containing processed files.
        time_series (bool): Flag denoting whether to load the time-series version of the data.

    Returns:
        train_features (pd.DataFrame): Training data features.
        train_labels (pd.DataFrame): Training data labels.
        test_features (pd.DataFrame): Testing data features.
        test_labels (pd.DataFrame): Testing data labels.
        test_scenarios (List[List[str]]): Scenarios included in the test dataset.
    """
    if time_series:
        try:
            test_features = pd.read_csv(f'{data_directory}/ts_test_features.csv')
            train_features = pd.read_csv(f'{data_directory}/ts_train_features.csv')
            test_labels = pd.read_csv(f'{data_directory}/ts_test_labels.csv')
            train_labels = pd.read_csv(f'{data_directory}/ts_train_labels.csv')
            test_scenarios = pd.read_csv(f'{data_directory}/ts_test_scenarios.csv').values.tolist()
        except FileNotFoundError:
                raise FileNotFoundError(f'Files not found at {data_directory}. Format must be in format \"ts_train_features.csv\"')
    else:
        try:
            test_features = pd.read_csv(f'{data_directory}/traditional_test_features.csv')
            train_features = pd.read_csv(f'{data_directory}/traditional_train_features.csv')
            test_labels = pd.read_csv(f'{data_directory}/traditional_test_labels.csv')
            train_labels = pd.read_csv(f'{data_directory}/traditional_train_labels.csv')
            test_scenarios = pd.read_csv(f'{data_directory}/traditional_test_scenarios.csv').values.tolist()
        except FileNotFoundError:
                raise FileNotFoundError(f'Files not found at {data_directory}. Format must be in format \"traditional_train_features.csv\"')
    
    return train_features, train_labels, test_features, test_labels, test_scenarios


def undummify(df, prefix_sep="-"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

def combine_testing_results(data_directory, preds, time_series=True, save_directory=None):
    emulator_data_args = _structure_emulatordata_args(input_args=None, time_series=time_series)
    
    emulator_data = EmulatorData(directory=data_directory)
    emulator_data, train_features, test_features, train_labels, test_labels = emulator_data.process(
        **emulator_data_args,
    )
    
    X_test = pd.DataFrame(emulator_data.unscale(values=test_features, values_type='inputs'))
    y_test = pd.Series(test_labels)
    
    test = X_test.drop(columns=[col for col in X_test.columns if 'lag' in col])
    test['true'] = y_test
    test['pred'] = preds
    test['mse'] = (test.true - test.pred)**2
    test['mae'] = abs(test.true - test.pred)
    
    test['sectors'] = round(test.sectors).astype(int)
    test['year'] = round(test.year).astype(int)
    
    test = undummify(test)
    
    if save_directory:
        if isinstance(save_directory, str):
            save_path = f"{save_directory}/results.csv"
            
        elif isinstance(save_directory, bool):
            save_path = f"results.csv"
        
        test.to_csv(save_path, index=False)
        
    return test

def group_by_run(dataset, column=None, condition=None,):
    modelnames = dataset.modelname.unique()
    exp_ids = dataset.exp_id.unique()
    sectors = dataset.sectors.unique()

    all_runs = [list(i) for i in list(product(modelnames, exp_ids, sectors))]

    all_trues = []
    all_preds = []
    scenarios = []
    for i, run in enumerate(all_runs):
        modelname = run[0]
        exp = run[1]
        sector = run[2]
        if column is None and condition is None:
            subset = dataset[(dataset.modelname == modelname) & (dataset.exp_id == exp) & (dataset.sectors == sector)]
        elif column is not None and condition is not None:
            subset = dataset[(dataset.modelname == modelname) & (dataset.exp_id == exp) & (dataset.sectors == sector) & (dataset[column] == condition)]
        else:
            raise ValueError('Column and condition type must be the same (None & None, not None & not None).')
        if not subset.empty:
            scenarios.append([])
            all_trues.append(subset.true.to_numpy())
            all_preds.append(subset.pred.to_numpy())
            
    return np.array(all_trues), np.array(all_preds), scenarios


def get_uncertainty_bands(data, confidence='95', quantiles=[0.05, 0.95]):
    z = {'95': 1.96, '99': 2.58}
    data = np.array(data)
    mean = data.mean(axis=0)
    sd = np.sqrt(data.var(axis=0))
    upper_ci = mean + (z[confidence] * (sd/np.sqrt(data.shape[0])))
    lower_ci = mean - (z[confidence] * (sd/np.sqrt(data.shape[0])))
    quantiles = np.quantile(data, quantiles, axis=0)
    upper_q = quantiles[1,:]
    lower_q = quantiles[0,:]
    return mean, sd, upper_ci, lower_ci, upper_q, lower_q

def create_distribution(year, dataset):
    data = dataset[:, year-2101] # -1 will be year 2100
    kde = gaussian_kde(data, bw_method='silverman')
    support = np.arange(-30, 20, 0.001)
    density = kde(support)
    return density, support

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def js_divergence(p, q):
    return jensenshannon(p, q)