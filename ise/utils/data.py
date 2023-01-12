"""Utility functions for handling data."""

import pandas as pd
from ise.utils.utils import _structure_emulatordata_args
from ise.data import EmulatorData
from itertools import product
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler
from typing import List



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
        try:
            test_features = pd.read_csv(f"{data_directory}/ts_test_features.csv")
            train_features = pd.read_csv(f"{data_directory}/ts_train_features.csv")
            test_labels = pd.read_csv(f"{data_directory}/ts_test_labels.csv")
            train_labels = pd.read_csv(f"{data_directory}/ts_train_labels.csv")
            test_scenarios = pd.read_csv(
                f"{data_directory}/ts_test_scenarios.csv"
            ).values.tolist()
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Files not found at {data_directory}. Format must be in format "ts_train_features.csv"'
            )
    else:
        try:
            test_features = pd.read_csv(
                f"{data_directory}/traditional_test_features.csv"
            )
            train_features = pd.read_csv(
                f"{data_directory}/traditional_train_features.csv"
            )
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


def combine_testing_results(
    data_directory: str,
    preds: np.ndarray, #|pd.Series|str,
    bounds: dict = None, #|pd.DataFrame = None,
    gp_data: dict = None, #|pd.DataFrame = None,
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

    test = undummify(test)
    test = unscale_column(test, column=['year', 'sector'])

    if bounds is not None:
        if not isinstance(bounds, pd.DataFrame):
            bounds = pd.DataFrame(bounds)
        # add bounds to dataframe
        test = test.merge(bounds, left_index=True, right_index=True)

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
        data (pd.DataFrame): Dataframe or array of NXM, typically from ise.utils.data.group_by_run.
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


def create_distribution(year: int, dataset: np.ndarray):
    """Creates a distribution from an array of numbers using a gaussian kernel density estimator.
    Takes an array and ensures it follows probability rules (e.g. integrate to 1, nonzero, etc.),
    useful for calculating divergences such as ise.utils.data.kl_divergence and ise.utils.data.js_divergence.

    Args:
        year (int): Year to generate the distribution.
        dataset (np.ndarray): MX85 matrix of true values or predictions for the series, see ise.utils.data.group_by_run.

    Returns:
        tuple: Tuple containing [density, support], or the output distribution and the x values associated with those probabilities.
    """
    data = dataset[:, year - 2101]  # -1 will be year 2100
    kde = gaussian_kde(data, bw_method="silverman")
    support = np.arange(-30, 20, 0.001)
    density = kde(support)
    return density, support


def kl_divergence(p: np.ndarray, q: np.ndarray):
    """Calculates the Kullback-Leibler Divergence between two distributions. Q is typically a
    'known' distirubtion and should be the true values, whereas P is typcically the test distribution,
    or the predicted distribution. Note the the KL divergence is assymetric, and near-zero values for
    p with a non-near zero values for q cause the KL divergence to inflate [citation].

    Args:
        p (np.ndarray): Test distribution
        q (np.ndarray): Known distribution

    Returns:
        float: KL Divergence
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def js_divergence(p: np.ndarray, q: np.ndarray):
    """Calculates the Jensen-Shannon Divergence between two distributions. Q is typically a
    'known' distirubtion and should be the true values, whereas P is typcically the test distribution,
    or the predicted distribution. Note the the JS divergence, unlike the KL divergence, is symetric.

    Args:
        p (np.ndarray): Test distribution
        q (np.ndarray): Known distribution

    Returns:
        float: JS Divergence
    """
    return jensenshannon(p, q)


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
        dataset["year"] = year_scaler.inverse_transform(
            np.array(dataset.year).reshape(-1, 1)
        )
        dataset["year"] = round(dataset.year).astype(int)

    return dataset
