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
    """Loads PyTorch model from saved state_dict.

    Args:
        model_path (str): Filepath to model state_dict.
        model_class (Model): Model class.
        architecture (dict): Defined architecture of pretrained model.
        mc_dropout (bool): Flag denoting wether the model was trained using MC Dropout.
        dropout_prob (float): Value between 0 and 1 denoting the dropout probability.

    Returns:
        model (Model): Pretrained model.
    """
    model = model_class(architecture, mc_dropout=mc_dropout, dropout_prob=dropout_prob)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


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
        all_files = [file for file in all_files if contains in file]

    if not_contains:
        all_files = [file for file in all_files if not_contains not in file]

    return all_files


def add_variable_to_nc(source_file_path, target_file_path, variable_name):
    """
    Copies a variable from a source NetCDF file to a target NetCDF file.

    Parameters:
    - source_file_path: Path to the source NetCDF file.
    - target_file_path: Path to the target NetCDF file.
    - variable_name: Name of the variable to be copied.

    Both files are assumed to have matching dimensions for the variable.
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
    """Loads training and testing data for machine learning models. These files are generated using
    functions in the ise.data.processing modules or process_data in the ise.pipelines.processing module.

    Args:
        data_directory (str): Directory containing processed files.
        time_series (bool): Flag denoting whether to load the time-series version of the data.

    Returns:
        tuple: Tuple containing [train features, train_labels, test_features, test_labels, test_scenarios], or the training and testing datasets including the scenarios used in testing.
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
    """Undummifies, or reverses pd.get_dummies, a dataframe. Includes taking encoded categorical
    variable columns (boolean indices), and converts them back into the original data format.

    Args:
        df (pd.DataFrame): Dataframe to be converted.
        prefix_sep (str, optional): Prefix separator used in pd.get_dummies. Recommended not to change this. Defaults to "-".

    Returns:
        _type_: _description_
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
    preds: np.ndarray,  # |pd.Series|str,
    sd: dict = None,  # |pd.DataFrame = None,
    gp_data: dict = None,  # |pd.DataFrame = None,
    time_series: bool = True,
    save_directory: str = None,
):
    """Creates testing results dataframe that reverts input data to original formatting and adds on
    predictions, losses, and uncertainty bounds. Useful for plotting purposes and overall analysis.

    Args:
        data_directory (str): Directory containing training and testing data.
        preds (np.ndarray | pd.Series | str): Array/Series of neural network predictions, or the path to the csv containing predictions.
        bounds (dict | pd.DataFrame): Dictionary or pd.DataFrame of uncertainty bounds to be added to the dataframe, generally outputted from ise.models.testing.pretrained.test_pretrained_model. Defaults to None.
        gp_data (dict | pd.DataFrame): Dictionary or pd.DataFrame containing gaussian process predictions to add to the dataset. Columns/keys must be `preds` and `std`. Defaults to None.
        time_series (bool, optional): Flag denoting whether to process the data as a time-series dataset or traditional non-time dataset. Defaults to True.
        save_directory (str, optional): Directory where output files will be saved. Defaults to None.

    Returns:
        pd.DataFrame: test results dataframe.
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
    """Groups the dataset into each individual simulation series by both the true value of the
    simulated SLE as well as the model predicted SLE. The resulting arrays are NXM matrices with
    N being the number of simulations and M being 85, or the length of the series.

    Args:
        dataset (pd.DataFrame): Dataset to be grouped
        column (str, optional): Column to subset on. Defaults to None.
        condition (str, optional): Condition to subset with. Can be int, str, float, etc. Defaults to None.

    Returns:
        tuple: Tuple containing [all_trues, all_preds], or NXM matrices of each series corresponding to true values and predicted values.
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
    """Calculates uncertainty bands on the monte carlo dropout protocol. Includes traditional
    confidence interval calculation as well as a quantile-based approach.

    Args:
        data (pd.DataFrame): Dataframe or array of NXM, typically from ise.utils.functions.group_by_run.
        confidence (str, optional): Confidence level, must be in [95, 99]. Defaults to '95'.
        quantiles (list[float], optional): Quantiles of uncertainty bands. Defaults to [0.05, 0.95].

    Returns:
        tuple: Tuple containing [mean, sd, upper_ci, lower_ci, upper_q, lower_q], or the mean prediction, standard deviation, and the lower and upper confidence interval and quantile bands.
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


def create_distribution(dataset: np.ndarray, min_range=-30, max_range=20, step=0.001):
    kde = gaussian_kde(dataset, bw_method="silverman")
    support = np.arange(min_range, max_range, step)
    density = kde(support)
    return density, support


def calculate_distribution_metrics(
    dataset: pd.DataFrame, column: str = None, condition: str = None
):
    """Wrapper for calculating distribution metrics from a dataset. Includes ise.utils.data.group_by_run to
    group the true values and predicted values into NXM matrices (with N=number of samples and
    M=85, or the number of years in the series). Then, it uses ise.utils.data.create_distribution to
    calculate individual distributions from the arrays and calculates the divergences.

    Args:
        dataset (pd.DataFrame): Dataset to be grouped
        column (str, optional): Column to subset on. Defaults to None.
        condition (str, optional): Condition to subset with. Can be int, str, float, etc. Defaults to None.

    Returns:
        dict: Dictionary containing dict['kl'] for the KL-Divergence and dict['js'] for the Jensen-Shannon Divergence.
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
    """Unscale column in dataset, particularly for unscaling year and sectors column given that
    they have a known range of values (2016-2100 and 1-18 respectively).

    Args:
        dataset (pd.DataFrame): Dataset containing columns to unscale.
        column (str | list, optional): Columns to be unscaled, must be in [year, sectors]. Can be both. Defaults to 'year'.

    Returns:
        pd.DataFrame: dataset containing unscaled columns.
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


"""Utility functions for handling various parts of the package, including argument checking and
formatting and file traversal."""


file_dir = os.path.dirname(os.path.realpath(__file__))


def check_input(input: str, options: List[str], argname: str = None):
    """Checks validity of input argument. Not used frequently due to error raising being better practice.

    Args:
        input (str): Input value.
        options (List[str]): Valid options for the input value.
        argname (str, optional): Name of the argument being tested. Defaults to None.
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
        all_files = [file for file in all_files if contains in file]

    if not_contains:
        all_files = [file for file in all_files if not_contains not in file]

    return all_files


def _structure_emulatordata_args(input_args: dict, time_series: bool):
    """Formats kwargs for EmulatorData processing. Includes establishing defaults if values are not
    supplied.

    Args:
        input_args (dict): Dictionary containin kwargs for EmulatorData.process()
        time_series (bool): Flag denoting whether the processing is time-series.

    Returns:
        dict: EmulatorData.process() kwargs formatted with defaults.
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
    """Formats the arguments for model architectures.

    Args:
        architecture (dict): User input for desired architecture.
        time_series (bool): Flag denoting whether to use time series model arguments or traditional.

    Returns:
        architecture (dict): Formatted architecture argument.
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
):
    if dataset_type.lower() == "sectors":
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

    return X, y


def to_tensor(x):
    """
    Converts input data to a PyTorch tensor of type float.

    Args:
        x: Input data to be converted. Must be a pandas dataframe, numpy array, or PyTorch tensor.

    Returns:
        A PyTorch tensor of type float.
    """
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
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
    Unscale the output data using the scaler saved during training.

    Args:
        y: Input data to be unscaled.
        scaler_path: Path to the scaler used for scaling the data.

    Returns:
        The unscaled data.
    """
    scaler = pkl.load(open(scaler_path, "rb"))
    y = scaler.inverse_transform(y)
    return y
