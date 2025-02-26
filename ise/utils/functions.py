import os
import pickle as pkl
from itertools import product
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler

from ise.evaluation.metrics import js_divergence, kl_divergence


def load_model(model_path, model_class, architecture, mc_dropout=False, dropout_prob=0.1):
    """
    Loads a PyTorch model from a saved state_dict file.

    Args:
        model_path (str): Path to the model's state dictionary file.
        model_class (type): Class reference of the model to be loaded.
        architecture (dict): Dictionary specifying the architecture of the model.
        mc_dropout (bool, optional): Whether the model uses Monte Carlo Dropout. Defaults to False.
        dropout_prob (float, optional): Dropout probability if MC Dropout is used. Defaults to 0.1.

    Returns:
        torch.nn.Module: The loaded PyTorch model set to the available device (CPU/GPU).
    """

    model = model_class(architecture, mc_dropout=mc_dropout, dropout_prob=dropout_prob)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


def get_all_filepaths(
    path: str, filetype: str = None, contains: str = None, not_contains: str = None
):
    """
    Retrieves all file paths in a directory, with optional filtering.

    Args:
        path (str): The directory path to search for files.
        filetype (str, optional): Filter files by extension (e.g., 'csv', 'nc'). Defaults to None.
        contains (str, optional): Only include files that contain this substring. Defaults to None.
        not_contains (str, optional): Exclude files that contain this substring. Defaults to None.

    Returns:
        List[str]: A list of file paths that match the specified criteria.
    """

    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        all_files += [os.path.join(dirpath, file) for file in filenames]

    if filetype:
        if filetype.lower() != "all":
            all_files = [file for file in all_files if file.endswith(filetype)]

    if contains:
        all_files = [file for file in all_files if contains in file]

    if not_contains:
        all_files = [file for file in all_files if not_contains not in file]

    return all_files


def add_variable_to_nc(source_file_path, target_file_path, variable_name):
    """
    Copies a variable from a source NetCDF file to a target NetCDF file.

    Args:
        source_file_path (str): Path to the source NetCDF file.
        target_file_path (str): Path to the target NetCDF file.
        variable_name (str): Name of the variable to be copied.

    Raises:
        FileNotFoundError: If the specified variable is not found in the source file.
    """

    # Open the source NetCDF file in read mode
    with Dataset(source_file_path, "r") as src_nc:
        # Check if the variable exists in the source file
        if variable_name in src_nc.variables:
            # Read the variable data and attributes
            variable_data = src_nc.variables[variable_name][:]
            variable_attributes = src_nc.variables[variable_name].ncattrs()

            # Open the target NetCDF file in append mode
            with Dataset(target_file_path, "a") as target_nc:
                # Create or overwrite the variable in the target file
                if variable_name in target_nc.variables:
                    print(
                        f"The '{variable_name}' variable already exists in the target file. Overwriting data."
                    )
                    target_nc.variables[variable_name][:] = variable_data
                else:
                    # Create the variable with the same datatype and dimensions
                    variable = target_nc.createVariable(
                        variable_name,
                        src_nc.variables[variable_name].datatype,
                        src_nc.variables[variable_name].dimensions,
                    )

                    # Copy the variable attributes
                    for attr_name in variable_attributes:
                        variable.setncattr(
                            attr_name, src_nc.variables[variable_name].getncattr(attr_name)
                        )

                    # Assign the data to the new variable
                    variable[:] = variable_data
        else:
            print(f"'{variable_name}' variable not found in the source file.")


def load_ml_data(data_directory: str, time_series: bool = True):
    """
    Loads machine learning training and testing data from CSV files.

    Args:
        data_directory (str): Directory containing the processed data files.
        time_series (bool, optional): Whether to load the time-series version of the data. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - train_features (pd.DataFrame): Training feature set.
            - train_labels (pd.Series): Training labels.
            - test_features (pd.DataFrame): Testing feature set.
            - test_labels (pd.Series): Testing labels.
            - test_scenarios (list): List of test scenarios.
    """

    if time_series:
        # TODO: Test scenarios has no use, get rid of it
        try:
            test_features = pd.read_csv(f"{data_directory}/ts_test_features.csv")
            train_features = pd.read_csv(f"{data_directory}/ts_train_features.csv")
            test_labels = pd.read_csv(f"{data_directory}/ts_test_labels.csv")
            train_labels = pd.read_csv(f"{data_directory}/ts_train_labels.csv")
            test_scenarios = pd.read_csv(f"{data_directory}/ts_test_scenarios.csv").values.tolist()
        except FileNotFoundError:
            try:
                test_features = pd.read_csv(f"{data_directory}/val_features.csv")
                train_features = pd.read_csv(f"{data_directory}/train_features.csv")
                test_labels = pd.read_csv(f"{data_directory}/val_labels.csv")
                train_labels = pd.read_csv(f"{data_directory}/train_labels.csv")
                test_scenarios = pd.read_csv(
                    f"{data_directory}/ts_test_scenarios.csv"
                ).values.tolist()
            except:
                raise FileNotFoundError(
                    f'Files not found at {data_directory}. Format must be in format "ts_train_features.csv"'
                )
    else:
        try:
            test_features = pd.read_csv(f"{data_directory}/traditional_test_features.csv")
            train_features = pd.read_csv(f"{data_directory}/traditional_train_features.csv")
            test_labels = pd.read_csv(f"{data_directory}/traditional_test_labels.csv")
            train_labels = pd.read_csv(f"{data_directory}/traditional_train_labels.csv")
            test_scenarios = pd.read_csv(
                f"{data_directory}/traditional_test_scenarios.csv"
            ).values.tolist()
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Files not found at {data_directory}. Format must be in format "traditional_train_features.csv"'
            )

    return (
        train_features,
        pd.Series(train_labels["sle"], name="sle"),
        test_features,
        pd.Series(test_labels["sle"], name="sle"),
        test_scenarios,
    )


def undummify(df: pd.DataFrame, prefix_sep: str = "-"):
    """
    Converts a one-hot encoded dataframe back to its categorical form.

    Args:
        df (pd.DataFrame): DataFrame containing one-hot encoded categorical columns.
        prefix_sep (str, optional): Separator used in column names to identify categories. Defaults to "-".

    Returns:
        pd.DataFrame: DataFrame with categorical values restored.
    """

    cols2collapse = {item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns}

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


def combine_testing_results(
    data_directory: str,
    preds: np.ndarray,
    sd: dict = None,
    gp_data: dict = None,
    time_series: bool = True,
    save_directory: str = None,
):
    """
    Combines test results into a DataFrame with predictions, uncertainties, and true values.

    Args:
        data_directory (str): Directory containing training and testing data.
        preds (np.ndarray | pd.Series | str): Predictions array, Series, or path to a CSV file with predictions.
        sd (dict | pd.DataFrame, optional): Standard deviations for uncertainty estimation. Defaults to None.
        gp_data (dict | pd.DataFrame, optional): Gaussian process predictions and standard deviations. Defaults to None.
        time_series (bool, optional): Whether to process time-series data. Defaults to True.
        save_directory (str, optional): Directory where results should be saved. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing test results with true values, predictions, errors, and uncertainty bounds.
    """


    (
        train_features,
        train_labels,
        test_features,
        test_labels,
        test_scenarios,
    ) = load_ml_data(data_directory, time_series=time_series)

    X_test = pd.DataFrame(test_features)
    if isinstance(test_labels, pd.Series):
        y_test = test_labels
    elif isinstance(test_labels, pd.DataFrame):
        y_test = pd.Series(test_labels["sle"])
    else:
        y_test = pd.Series(test_labels)

    test = X_test.drop(columns=[col for col in X_test.columns if "lag" in col])
    test["true"] = y_test
    test["pred"] = np.array(pd.read_csv(preds)) if isinstance(preds, str) else preds
    test["mse"] = (test.true - test.pred) ** 2
    test["mae"] = abs(test.true - test.pred)

    if gp_data:
        test["gp_preds"] = gp_data["preds"]
        test["gp_std"] = gp_data["std"]
        test["gp_upper_bound"] = test.gp_preds + 1.96 * test.gp_std
        test["gp_lower_bound"] = test.gp_preds - 1.96 * test.gp_std

    test = undummify(test)
    test = unscale_column(test, column=["year", "sector"])

    if sd is not None:
        test["sd"] = sd
        test["upper_bound"] = preds + 1.96 * sd
        test["lower_bound"] = preds - 1.96 * sd

    if save_directory:
        if isinstance(save_directory, str):
            save_path = f"{save_directory}/results.csv"

        elif isinstance(save_directory, bool):
            save_path = f"results.csv"

        test.to_csv(save_path, index=False)

    return test


def group_by_run(
    dataset: pd.DataFrame,
    column: str = None,
    condition: str = None,
):
    """
    Groups dataset simulations into structured matrices for true and predicted values.

    Args:
        dataset (pd.DataFrame): Dataset containing simulation results.
        column (str, optional): Column name to subset on. Defaults to None.
        condition (str, optional): Condition for filtering the dataset. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - all_trues (np.ndarray): Matrix of true values (N x M, where N is the number of simulations and M is time steps).
            - all_preds (np.ndarray): Matrix of predicted values.
            - scenarios (list): List of scenario information for each simulation.
    """


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
            subset = dataset[
                (dataset.modelname == modelname)
                & (dataset.exp_id == exp)
                & (dataset.sectors == sector)
            ]
        elif column is not None and condition is not None:
            subset = dataset[
                (dataset.modelname == modelname)
                & (dataset.exp_id == exp)
                & (dataset.sectors == sector)
                & (dataset[column] == condition)
            ]
        else:
            raise ValueError(
                "Column and condition type must be the same (None & None, not None & not None)."
            )
        if not subset.empty:
            scenarios.append([modelname, exp, sector])
            all_trues.append(subset.true.to_numpy())
            all_preds.append(subset.pred.to_numpy())

    return np.array(all_trues), np.array(all_preds), scenarios


def get_uncertainty_bands(
    data: pd.DataFrame, confidence: str = "95", quantiles: List[float] = [0.05, 0.95]
):
    """
    Computes uncertainty bands using confidence intervals and quantiles.

    Args:
        data (pd.DataFrame): Data matrix of shape (N, M), where N is samples and M is time steps.
        confidence (str, optional): Confidence level ('95' or '99'). Defaults to "95".
        quantiles (List[float], optional): Quantiles for uncertainty bands. Defaults to [0.05, 0.95].

    Returns:
        tuple: A tuple containing:
            - mean (np.ndarray): Mean values.
            - sd (np.ndarray): Standard deviation values.
            - upper_ci (np.ndarray): Upper confidence interval.
            - lower_ci (np.ndarray): Lower confidence interval.
            - upper_q (np.ndarray): Upper quantile bound.
            - lower_q (np.ndarray): Lower quantile bound.
    """

    z = {"95": 1.96, "99": 2.58}
    data = np.array(data)
    mean = data.mean(axis=0)
    sd = np.sqrt(data.var(axis=0))
    upper_ci = mean + (z[confidence] * (sd / np.sqrt(data.shape[0])))
    lower_ci = mean - (z[confidence] * (sd / np.sqrt(data.shape[0])))
    quantiles = np.quantile(data, quantiles, axis=0)
    upper_q = quantiles[1, :]
    lower_q = quantiles[0, :]
    return mean, sd, upper_ci, lower_ci, upper_q, lower_q


def create_distribution(dataset: np.ndarray, min_range=-30, max_range=20, step=0.01):
    """
    Creates a probability density function (PDF) using Gaussian kernel density estimation (KDE).

    Args:
        dataset (np.ndarray): Input data for KDE.
        min_range (float, optional): Minimum range for support values. Defaults to -30.
        max_range (float, optional): Maximum range for support values. Defaults to 20.
        step (float, optional): Step size for the support values. Defaults to 0.01.

    Returns:
        tuple: A tuple containing:
            - density (np.ndarray): Density values from KDE.
            - support (np.ndarray): Support values for the density function.
    """

    kde = gaussian_kde(dataset, bw_method="silverman")
    support = np.arange(min_range, max_range, step)
    density = kde(support)
    return density, support


def calculate_distribution_metrics(
    dataset: pd.DataFrame, column: str = None, condition: str = None
):
    """
    Computes distribution divergence metrics between true and predicted values.

    This function groups the dataset by simulation runs, creates probability 
    distributions for true and predicted values, and calculates the 
    Kullback-Leibler (KL) and Jensen-Shannon (JS) divergences.

    Args:
        dataset (pd.DataFrame): The dataset containing true and predicted values.
        column (str, optional): Column name to subset on. Defaults to None.
        condition (str, optional): Value to filter the dataset based on the specified column. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - 'kl' (float): KL-Divergence value.
            - 'js' (float): Jensen-Shannon Divergence value.
    """

    trues, preds, _ = group_by_run(dataset, column=column, condition=condition)
    true_distribution, _ = create_distribution(year=2100, dataset=trues)
    pred_distribution, _ = create_distribution(year=2100, dataset=preds)
    distribution_metrics = {
        "kl": kl_divergence(pred_distribution, true_distribution),
        "js": js_divergence(pred_distribution, true_distribution),
    }
    return distribution_metrics


def unscale_column(dataset: pd.DataFrame, column: str = "year"):
    """
    Unscales specified columns back to their original range using known value distributions.

    This function is specifically used to revert the normalization of 'year' and 
    'sectors' columns since they have known value ranges.

    Args:
        dataset (pd.DataFrame): Dataset containing the scaled columns.
        column (str or list, optional): Column(s) to be unscaled. 
            Can be 'year', 'sectors', or a list containing both. Defaults to "year".

    Returns:
        pd.DataFrame: Dataset with the specified column(s) unscaled.
    """


    if isinstance(column, str):
        column = [column]

    if "sectors" in column:
        sectors_scaler = MinMaxScaler().fit(np.arange(1, 19).reshape(-1, 1))
        dataset["sectors"] = sectors_scaler.inverse_transform(
            np.array(dataset.sectors).reshape(-1, 1)
        )
        dataset["sectors"] = round(dataset.sectors).astype(int)

    if "year" in column:
        year_scaler = MinMaxScaler().fit(np.arange(2016, 2101).reshape(-1, 1))
        dataset["year"] = year_scaler.inverse_transform(np.array(dataset.year).reshape(-1, 1))
        dataset["year"] = round(dataset.year).astype(int)

    return dataset

file_dir = os.path.dirname(os.path.realpath(__file__))


def check_input(input: str, options: List[str], argname: str = None):
    """
    Validates whether a given input string is within an expected list of options.

    Args:
        input (str): The input value to validate.
        options (List[str]): A list of valid options.
        argname (str, optional): Name of the argument being checked for better error messaging. Defaults to None.

    Raises:
        ValueError: If the input is not in the list of allowed options.
    """

    # simple assert that input is in the designated options (readability purposes only)
    if isinstance(input, str):
        input = input.lower()
    if input not in options:
        if argname is not None:
            raise ValueError(f"{argname} must be in {options}, received {input}")
        raise ValueError(f"input must be in {options}, received {input}")


def get_all_filepaths(
    path: str, filetype: str = None, contains: str = None, not_contains: str = None
):
    """Retrieves all filepaths for files within a directory. Supports subsetting based on filetype
    and substring search.

    Args:
        path (str): Path to directory to be searched.
        filetype (str, optional): File type to be returned (e.g. csv, nc). Defaults to None.
        contains (str, optional): Substring that files found must contain. Defaults to None.
        not_contains(str, optional): Substring that files found must NOT contain. Defaults to None.

    Returns:
        List[str]: list of files within the directory matching the input criteria.
    """
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        all_files += [os.path.join(dirpath, file) for file in filenames]

    if filetype:
        if filetype.lower() != "all":
            all_files = [file for file in all_files if file.endswith(filetype)]

    if contains:
        if isinstance(contains, str):
            contains = [contains]
        for substr in contains:
            all_files = [file for file in all_files if substr in file]

    if not_contains:
        if isinstance(not_contains, str):
            not_contains = [not_contains]
        for substr in not_contains:
            all_files = [file for file in all_files if substr not in file]

    return all_files


def _structure_emulatordata_args(input_args: dict, time_series: bool):
    """
    Formats the keyword arguments for processing EmulatorData, applying default values if needed.

    Args:
        input_args (dict): Dictionary of keyword arguments for EmulatorData processing.
        time_series (bool): Whether the processing is for time-series data.

    Returns:
        dict: A dictionary of EmulatorData processing arguments with applied defaults.
    """

    emulator_data_defaults = dict(
        target_column="sle",
        drop_missing=True,
        drop_columns=["groupname", "experiment"],
        boolean_indices=True,
        scale=True,
        split_type="batch",
        drop_outliers="explicit",
        drop_expression=[("sle", "<", -26.3)],
        time_series=time_series,
        lag=None,
    )

    if time_series:
        emulator_data_defaults["lag"] = 5

    # If no other args are supplied, use defaults
    if input_args is None:
        return emulator_data_defaults
    # else, replace provided key value pairs in the default dict and reassign
    else:
        for key in input_args.keys():
            emulator_data_defaults[key] = input_args[key]
        output_args = emulator_data_defaults

    return output_args


def _structure_architecture_args(architecture, time_series):
    """
    Structures the architecture arguments for model training.

    Ensures that the provided architecture dictionary is appropriate for either 
    time-series or traditional models.

    Args:
        architecture (dict): Dictionary specifying model architecture parameters.
        time_series (bool): Whether the model is for time-series data.

    Returns:
        dict: A structured architecture dictionary.

    Raises:
        AttributeError: If incompatible architecture arguments are used.
    """


    # Check to make sure inappropriate args are not used
    if not time_series and (
        "num_rnn_layers" in architecture.keys() or "num_rnn_hidden" in architecture.keys()
    ):
        raise AttributeError(
            f"Time series architecture args must be in [num_linear_layers, nodes], received {architecture}"
        )
    if time_series and (
        "nodes" in architecture.keys() or "num_linear_layers" in architecture.keys()
    ):
        raise AttributeError(
            f"Time series architecture args must be in [num_rnn_layers, num_rnn_hidden], received {architecture}"
        )

    if architecture is None:
        if time_series:
            architecture = {
                "num_rnn_layers": 3,
                "num_rnn_hidden": 128,
            }
        else:
            architecture = {
                "num_linear_layers": 4,
                "nodes": [128, 64, 32, 1],
            }
    else:
        return architecture
    return architecture


def get_X_y(
    data,
    dataset_type="sectors",
    return_format=None,
    cols="all",
    with_chars=True,
):  
    """
    Extracts input features (X) and target labels (y) from a dataset.

    Supports various dataset types (sectors, regions, scenarios) and formats 
    (numpy, tensor, pandas).

    Args:
        data (str or pd.DataFrame): Filepath to the dataset CSV or a pandas DataFrame.
        dataset_type (str, optional): The type of dataset ('sectors', 'regions', 'scenarios'). Defaults to "sectors".
        return_format (str, optional): Format of the returned data ('numpy', 'tensor', or 'pandas'). Defaults to None.
        cols (str or list, optional): Columns to include in the features. Defaults to "all".
        with_chars (bool, optional): Whether to include characteristic columns in features. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame or np.ndarray or torch.Tensor): The input features.
            - y (pd.DataFrame or np.ndarray or torch.Tensor): The target labels.
            - scenarios (list, optional): Scenario identifiers if dataset type is "regions".
    """

    if isinstance(data, str):
        data = pd.read_csv(data)
    
    ice_sheet = "AIS" if dataset_type.lower() == "regions" and 'smb_anomaly' in data.columns else 'GrIS'
    regions = True if dataset_type.lower() == "regions" else False
    if dataset_type.lower() == "sectors" or regions:
        
        if regions:
            sector_to_region = {}
            if ice_sheet == 'AIS':
                for i, sector in enumerate(data.sector.unique()):
                    if i > 0 and i < 6:
                        sector_to_region[sector] = 1
                    elif i > 5 and i < 17:
                        sector_to_region[sector] = 2
                    else:
                        sector_to_region[sector] = 3
            else:
                for i, sector in enumerate(data.sector.unique()):
                    sector_to_region[sector] = 1
            
            data['region'] = data.sector.map(sector_to_region)
            data['id'] = data['id'].apply(lambda x: "_".join(x.split('_')[:-1]))
            data = data.groupby(['region', 'id', 'year', 'Scenario']).agg({
                'sle': 'mean',   # for GP, used mean instead of sum so that scales of errors are the same
                **{col: 'mean' for col in data.columns.difference(['sle']) if data[col].dtype == 'float64'}
            }, )
            data = data.drop(columns=['year']) # drop year so that there aren't two year columns after reset_index()
            data = data.reset_index()
            scenarios = data.Scenario
            
        dropped_columns = [
            "id",
            "cmip_model",
            "pathway",
            "exp",
            "ice_sheet",
            "Scenario",
            "Ocean forcing",
            "Ocean sensitivity",
            "Ice shelf fracture",
            "Tier",
            "aogcm",
            "id",
            "exp",
            "model",
            "ivaf",
            "outlier",
        ]
        dropped_columns = [x for x in data.columns if x in dropped_columns]
        X_drop = [x for x in data.columns if "sle" in x] + dropped_columns
        X = data.drop(columns=X_drop)
        y = data[[x for x in data.columns if "sle" in x]]
    
    elif 'scenario' in dataset_type.lower():
        dropped_columns = [
            'sector', 
            'year',
            "cmip_model",
            "pathway",
            "exp",
            "ice_sheet",
            "Ocean forcing",
            "Ocean sensitivity",
            "Ice shelf fracture",
            "Tier",
            "aogcm",
            "id",
            "exp",
            "model",
            "ivaf",
            "outlier",
            'sle',
        ]
        dropped_columns = [x for x in data.columns if x in dropped_columns] + [x for x in data.columns if 'lag' in x]
        X_drop = [x for x in data.columns if "Scenario" in x] + dropped_columns
        X = data.drop(columns=X_drop)
        y = data[[x for x in data.columns if "Scenario" in x]]
        
    if isinstance(cols, list) and cols:
        X = X[list(cols)]
        
    if not with_chars:
        char_cols = ['initial', 'numerics', 'ice_flow', 'initialization', 'velocity', 'bed', 
                     'surface', 'ghf', 'res', 'Ocean', 'Ice shelf', 'stress', 'resolution', 'init', 
                     'melt', 'ice_front', 'open', 'standard', ]
        drop_cols = [col for col in X.columns if any(substring in col for substring in char_cols)]
        X = X.drop(columns=drop_cols)
        
    if return_format is not None:
        if return_format.lower() == "numpy":
            return X.values, y.values
        elif return_format.lower() == "tensor":
            return torch.tensor(X.values), torch.tensor(y.values)
        elif return_format.lower() == "pandas":
            pass
        else:
            raise ValueError(
                f"return_format must be in ['numpy', 'tensor', 'pandas'], received {return_format}"
            )
    
    if regions:
        return X, y, scenarios

    return X, y


def to_tensor(x):
    """
    Converts input data into a PyTorch tensor with float32 dtype.

    Args:
        x (pd.DataFrame, np.ndarray, or torch.Tensor): Input data.

    Returns:
        torch.Tensor: Converted tensor.

    Raises:
        ValueError: If the input data type is not supported.
    """

    if x is None:
        return None
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = torch.tensor(x.values, dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        raise ValueError("Data must be a pandas dataframe, numpy array, or PyTorch tensor")
    return x.float()


def unscale(y, scaler_path):
    """
    Unscales a dataset using a previously saved MinMaxScaler.

    Args:
        y (np.ndarray): The scaled data.
        scaler_path (str): Path to the saved MinMaxScaler object.

    Returns:
        np.ndarray: The unscaled data.
    """

    scaler = pkl.load(open(scaler_path, "rb"))
    y = scaler.inverse_transform(y)
    return y

def get_data(data_dir, dataset_type='sectors', return_format='tensor'):
    """
    Loads training, validation, and test datasets, formatting them for model training.

    Args:
        data_dir (str): Path to the directory containing the dataset files.
        dataset_type (str, optional): Type of dataset ('sectors' or 'scenarios'). Defaults to 'sectors'.
        return_format (str, optional): Format of the returned data ('tensor', 'numpy', or 'pandas'). Defaults to 'tensor'.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame, np.ndarray, or torch.Tensor): Training features.
            - y_train (pd.DataFrame, np.ndarray, or torch.Tensor): Training labels.
            - X_val (pd.DataFrame, np.ndarray, or torch.Tensor): Validation features.
            - y_val (pd.DataFrame, np.ndarray, or torch.Tensor): Validation labels.
            - X_test (pd.DataFrame, np.ndarray, or torch.Tensor): Testing features.
            - y_test (pd.DataFrame, np.ndarray, or torch.Tensor): Testing labels.
    """

    X_train, y_train = get_X_y(f"{data_dir}/train.csv", dataset_type=dataset_type, return_format=return_format)
    X_val, y_val = get_X_y(f"{data_dir}/val.csv", dataset_type=dataset_type, return_format=return_format)
    X_test, y_test = get_X_y(f"{data_dir}/test.csv", dataset_type=dataset_type, return_format=return_format)
    return X_train, y_train, X_val, y_val, X_test, y_test