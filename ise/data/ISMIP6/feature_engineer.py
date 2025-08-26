import json
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm
import warnings


class FeatureEngineer:
    """
    A class for performing feature engineering on a given dataset, including preprocessing,
    scaling, dataset splitting, and outlier handling.

    Args:
        ice_sheet (str): The name of the ice sheet being analyzed.
        data (pd.DataFrame): The input dataset.
        fill_mrro_nans (bool, optional): Whether to fill missing values in the 'mrro' column. Defaults to False.
        split_dataset (bool, optional): Whether to split the dataset into training, validation, and test sets. Defaults to False.
        train_size (float, optional): Proportion of data to use for training. Defaults to 0.7.
        val_size (float, optional): Proportion of data to use for validation. Defaults to 0.15.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.15.
        output_directory (str, optional): Directory to save the split datasets. Defaults to None.

    Attributes:
        data (pd.DataFrame): The input dataset.
        train_size (float): Proportion of training data.
        val_size (float): Proportion of validation data.
        test_size (float): Proportion of testing data.
        output_directory (str): Directory to save datasets.
        scaler_X_path (str): Path to the saved input feature scaler.
        scaler_y_path (str): Path to the saved target variable scaler.
        scaler_X (scaler object): Scaler for input features.
        scaler_y (scaler object): Scaler for target variables.
        train (pd.DataFrame): Training dataset.
        val (pd.DataFrame): Validation dataset.
        test (pd.DataFrame): Test dataset.
        _including_model_characteristics (bool): Whether model characteristics have been included.

    Methods:
        split_data: Splits dataset into train, validation, and test sets.
        fill_mrro_nans: Fills missing values in the 'mrro' column.
        scale_data: Scales input and target variables using a specified method.
        unscale_data: Reverses the scaling transformation.
        add_lag_variables: Adds lag features to the dataset.
        backfill_outliers: Replaces extreme values in target variables.
        drop_outliers: Removes outliers based on specified criteria.
        add_model_characteristics: Merges model characteristics into the dataset.
    """

    def __init__(
        self,
        ice_sheet,
        data: pd.DataFrame,
        fill_mrro_nans: bool = False,
        split_dataset: bool = False,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        output_directory: str = None,
    ):
        self.data = data
        try:
            self.data = self.data.sort_values(by=["model", "exp", "sector", "year"])
        except:
            pass
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.output_directory = output_directory
        self.ice_sheet = ice_sheet

        self.scaler_X_path = None
        self.scaler_y_path = None
        self.scaler_X = None
        self.scaler_y = None

        if fill_mrro_nans:
            self.data = self.fill_mrro_nans(method="zero")

        if split_dataset:
            self.train, self.val, self.test = self.split_data(
                data, train_size, val_size, test_size, output_directory, random_state=42
            )
        self._including_model_characteristics = False

        self.train = None
        self.val = None
        self.test = None

    def split_data(
        self,
        data=None,
        train_size=None,
        val_size=None,
        test_size=None,
        output_directory=None,
        random_state=42,
    ):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            data (pd.DataFrame, optional): The input dataset. Defaults to None.
            train_size (float, optional): Proportion of training data. Defaults to None.
            val_size (float, optional): Proportion of validation data. Defaults to None.
            test_size (float, optional): Proportion of testing data. Defaults to None.
            output_directory (str, optional): Directory to save split datasets. Defaults to None.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            tuple: Training, validation, and test datasets as pandas DataFrames.
        """
        if data is not None:
            self.data = data
        if train_size is not None:
            self.train_size = train_size
        if val_size is not None:
            self.val_size = val_size
        if output_directory is not None:
            self.output_directory = output_directory

        self.train, self.val, self.test = split_training_data(
            self.data,
            self.train_size,
            self.val_size,
            self.test_size,
            self.output_directory,
            random_state,
        )
        return self.train, self.val, self.test

    def fill_mrro_nans(self, method, data=None):
        """
        Fills missing values in the 'mrro' column.

        Args:
            method (str): The method used to fill missing values.
            data (pd.DataFrame, optional): The dataset. Defaults to None.

        Returns:
            pd.DataFrame: The dataset with missing values filled.
        """
        if data is not None:
            self.data = data
        if "mrro_anomaly" not in self.data.columns:
            print('mrro_anomaly not in columns, skipping fill_mrro_nans()')
            return self.data
        
        self.data = fill_mrro_nans(self.data, method)

        return self.data

    def scale_data(self, X=None, y=None, method="standard", save_dir=None):
        """
        Scales input (X) and target (y) variables using a specified scaling method.

        Args:
            X (pd.DataFrame or np.ndarray, optional): Input data. Defaults to None.
            y (pd.DataFrame or np.ndarray, optional): Target data. Defaults to None.
            method (str, optional): Scaling method ('standard', 'minmax', 'robust'). Defaults to 'standard'.
            save_dir (str, optional): Directory to save scalers. Defaults to None.

        Returns:
            tuple: Scaled X and y values.
        """
        
        if X is not None:
            self.X = X
        else:
            if self._including_model_characteristics:
                dropped_columns = [
                    "id",
                    "cmip_model",
                    "pathway",
                    "exp",
                    "ice_sheet",
                    "Scenario",
                    "Tier",
                    "aogcm",
                    "id",
                    "exp",
                    "model",
                    "ivaf",
                    "year"
                ] + list(self.data.columns[self.data.dtypes == bool])
            else:
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
                    "year",
                ]
            dropped_columns = [x for x in self.data.columns if x in dropped_columns]
            dropped_data = self.data[dropped_columns]
            self.X = self.data.drop(
                columns=[x for x in self.data.columns if "sle" in x] + dropped_columns
            )

        if y is not None:
            self.y = y
        else:
            self.y = self.data[[x for x in self.data.columns if "sle" in x]]

        if self.scaler_X_path is not None and self.scaler_y_path is not None:
            scaler_X = pickle.load(open(self.scaler_X_path, "rb"))
            scaler_y = pickle.load(open(self.scaler_y_path, "rb"))

            return scaler_X.transform(self.X), scaler_y.transform(self.y)
        elif self.scaler_X is not None and self.scaler_y is not None:
            return self.scaler_X.transform(self.X), self.scaler_y.transform(self.y)

        if (self.X is None and X is None) or (self.y is None and y is None):
            raise ValueError(
                "X and y must be provided if they are not already stored in the class instance."
            )

        # Initialize the scalers based on the method
        if method == "standard":
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif method == "minmax":
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        elif method == "robust":
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
        else:
            raise ValueError("method must be 'standard', 'minmax', or 'robust'")

        # Store scalers in the class instance for potential future use
        self.scaler_X, self.scaler_y = scaler_X, scaler_y

        # Fit and transform X
        if isinstance(self.X, pd.DataFrame):
            X_data = self.X.values
        elif isinstance(self.X, np.ndarray):
            X_data = self.X
        else:
            raise TypeError("X must be either a pandas DataFrame or a NumPy array.")

        scaler_X.fit(X_data)
        X_scaled = scaler_X.transform(X_data)
        
        # categorical_cols = [x for x in self.X.columns if len(set(self.X[x])) <= 2]
        # self.X[categorical_cols] = self.X[categorical_cols].astype('category')

        # Fit and transform y
        if isinstance(self.y, pd.DataFrame):
            y_data = self.y.values
        elif isinstance(self.y, np.ndarray):
            y_data = self.y
        else:
            raise TypeError("X must be either a pandas DataFrame or a NumPy array.")

        scaler_y.fit(y_data)
        y_scaled = scaler_y.transform(y_data)
        self.scaler_X, self.scaler_y = scaler_X, scaler_y

        # Optionally save the scalers
        if save_dir is not None:
            if os.path.exists(f"{save_dir}/scalers/"):
                self.scaler_X_path = f"{save_dir}/scalers/scaler_X.pkl"
                self.scaler_y_path = f"{save_dir}/scalers/scaler_y.pkl"
            else:
                self.scaler_X_path = f"{save_dir}/scaler_X.pkl"
                self.scaler_y_path = f"{save_dir}/scaler_y.pkl"
            with open(self.scaler_X_path, "wb") as f:
                pickle.dump(scaler_X, f)
            with open(self.scaler_y_path, "wb") as f:
                pickle.dump(scaler_y, f)

        self.data = pd.concat(
            [
                pd.DataFrame(X_scaled, columns=self.X.columns, index=self.X.index),
                pd.DataFrame(y_scaled, columns=self.y.columns, index=self.y.index),
                dropped_data,
            ],
            axis=1,
        )

        return X_scaled, y_scaled

    def unscale_data(self, X=None, y=None, scaler_X_path=None, scaler_y_path=None):
        """
        Reverses the scaling transformation for input (X) and target (y) variables.

        Args:
            X (pd.DataFrame or np.ndarray, optional): The input data to be unscaled. Defaults to None.
            y (pd.DataFrame, np.ndarray, or torch.Tensor, optional): The target data to be unscaled. Defaults to None.
            scaler_X_path (str, optional): Path to the stored input scaler. Defaults to None.
            scaler_y_path (str, optional): Path to the stored target scaler. Defaults to None.

        Returns:
            tuple: Unscaled X and y data.
        """
        if scaler_X_path is not None:
            self.scaler_X_path = scaler_X_path
        if scaler_y_path is not None:
            self.scaler_y_path = scaler_y_path

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # Load scaler for X
        if X is not None:
            if self.scaler_X_path is None and self.scaler_X is None:
                raise ValueError("scaler_X_path must be provided if X is not None and self.scaler_X is None.")
            if self.scaler_X is None:
                with open(self.scaler_X_path, "rb") as f:
                    scaler_X = pickle.load(f)
            else:
                scaler_X = self.scaler_X
            X_unscaled = scaler_X.inverse_transform(X)
            if isinstance(X, pd.DataFrame):
                X_unscaled = pd.DataFrame(X_unscaled, columns=X.columns, index=X.index)
        else:
            X_unscaled = None

        # Load scaler for y
        if y is not None:
            if self.scaler_y_path is None and self.scaler_y is None:
                raise ValueError("scaler_y_path must be provided if y is not None and self.scaler_y is None.")
            if self.scaler_y is None:
                with open(self.scaler_y_path, "rb") as f:
                    scaler_y = pickle.load(f)
            else:
                scaler_y = self.scaler_y
            y_unscaled = scaler_y.inverse_transform(y)
            if isinstance(y, pd.DataFrame):
                y_unscaled = pd.DataFrame(y_unscaled, columns=y.columns, index=y.index)
        else:
            y_unscaled = None

        return X_unscaled, y_unscaled

    def add_lag_variables(self, lag, data=None):
        """
        Adds lagged versions of predictor variables to the dataset.

        Args:
            lag (int): Number of time steps to lag the variables.
            data (pd.DataFrame, optional): The dataset. If not provided, the class attribute 'data' is used.

        Returns:
            FeatureEngineer: The modified instance with lag variables added.
        """
        if data is not None:
            self.data = data
        self.data = add_lag_variables(self.data, lag)
        return self

    def backfill_outliers(self, percentile=99.999, data=None):
        """
        Replaces extreme values in target variables with the previous row's value.

        Args:
            percentile (float, optional): Percentile threshold for identifying outliers. Defaults to 99.999.
            data (pd.DataFrame, optional): The dataset. If not provided, the class attribute 'data' is used.

        Returns:
            FeatureEngineer: The modified instance with outliers handled.
        """
        if data is not None:
            self.data = data
        self.data = backfill_outliers(self.data, percentile=percentile)
        return self

    def drop_outliers(self, method, column, expression=None, quantiles=[0.01, 0.99], data=None):
        """
        Drops simulations that are outliers based on the provided method.

        Args:
            method (str): Method of outlier deletion ('quantile' or 'explicit').
            column (str): Column used for detecting outliers.
            expression (list[tuple], optional): List of filtering expressions in the form [(column, operator, value)]. Defaults to None.
            quantiles (list[float], optional): Quantiles for 'quantile' method. Defaults to [0.01, 0.99].
            data (pd.DataFrame, optional): The dataset. If not provided, the class attribute 'data' is used.

        Returns:
            FeatureEngineer: The modified instance with outliers removed.
        """
        if data is not None:
            self.data = data
        self.data = drop_outliers(self.data, column, method, expression, quantiles)
        return self

    def add_model_characteristics(
        self,
        data=None,
        model_char_path=None,
        encode=True,
        ids_path=None,
    ):
        """
        Merges model characteristic data with the dataset.

        Args:
            data (pd.DataFrame, optional): The dataset. If not provided, the class attribute 'data' is used.
            model_char_path (str, optional): Path to the model characteristics file. Defaults to the internal path.
            encode (bool, optional): Whether to one-hot encode categorical characteristics. Defaults to True.
            ids_path (str, optional): Path to an additional ID mapping file. Defaults to None.

        Returns:
            FeatureEngineer: The modified instance with model characteristics added.
        """
        
        if data is not None:
            self.data = data
        if model_char_path is None:
            model_char_path = f"./ise/utils/data_files/{self.ice_sheet}_model_characteristics.csv"
        self.data = add_model_characteristics(self.data, model_char_path, encode, ids_path=ids_path)
        self._including_model_characteristics = True

        return self

    def exclude_fetish_models(self, data=None, exclude='both'):
        """
        Excludes specific models from the dataset.

        Args:
            data (pd.DataFrame, optional): The dataset. If not provided, the class attribute 'data' is used.

        Returns:
            FeatureEngineer: The modified instance with specific models excluded.
        """
        if data is not None:
            self.data = data
        self.data = exclude_fetish_models(self.data, exclude)
        return self


def scale_data(data, scaler_path, ):
    """
    Scales the provided dataset using a pre-trained scaler.

    Args:
        data (pd.DataFrame): The dataset to be scaled.
        scaler_path (str): Path to the saved scaler.

    Returns:
        pd.DataFrame: The scaled dataset.
    """
    dropped_columns = [
        "id",
        "cmip_model",
        "pathway",
        "exp",
        "ice_sheet",
        "Scenario",
        "Tier",
        "aogcm",
        "id",
        "exp",
        "model",
        "ivaf",
    ]

    dropped_columns = [x for x in data.columns if x in dropped_columns]
    dropped_data = data[dropped_columns]
    data = data.drop(
        columns=[x for x in data.columns if "sle" in x] + dropped_columns
    )
    cols = data.columns

    scaler = pickle.load(open(scaler_path, "rb"))
    scaled = scaler.transform(data)
    scaled = pd.DataFrame(scaled, columns=cols,)
    if 'outlier' in scaled.columns:
        scaled = scaled.drop(columns=['outlier'])
    return scaled

def add_model_characteristics(
    data, model_char_path=r"./ise/utils/data_files/model_characteristics.csv", encode=True, ids_path=None
) -> pd.DataFrame:
    """
    Adds model characteristics to the dataset.

    Args:
        data (pd.DataFrame): The input dataset.
        model_char_path (str, optional): Path to the model characteristics file. Defaults to internal path.
        encode (bool, optional): Whether to one-hot encode categorical characteristics. Defaults to True.
        ids_path (str, optional): Path to an additional ID mapping file. Defaults to None.

    Returns:
        pd.DataFrame: The dataset with model characteristics added.
    """
    model_chars = pd.read_csv(model_char_path)
    all_data = pd.merge(data, model_chars, on="model", how="left")
    existing_char_columns = [
        "Ocean forcing",
        "Ocean sensitivity",
        "Ice shelf fracture",
    ]  # These are the columns that are already in the data and should not be encoded

    # if 'Ocean forcing' not in data.columns:
    #     if ids_path is None:
    #         raise ValueError("ids must be provided if 'Ocean forcing' is not in the data.")
    #     else:
    #         ids = json.load(open(ids_path, 'r'))

    if encode:
        all_data = pd.get_dummies(
            all_data,
            columns=[
                x
                for x in model_chars.columns
                if x
                not in [
                    "initial_year",
                    "model",
                    "Scenario",
                ]
            ]
            + existing_char_columns,
        )

    return all_data


def backfill_outliers(data, percentile=99.999):
    """
    Replaces extreme values in y-values (above the specified percentile and below the 1-percentile across all y-values)
    with the value from the previous row.

    Args:
        data (pd.DataFrame): The dataset containing y-values.
        percentile (float, optional): The percentile threshold to define upper extreme values. Defaults to 99.999.

    Returns:
        pd.DataFrame: The dataset with extreme values replaced using backfill.
    """

    # Assuming y-values are in columns named with 'sle' as mentioned in other methods
    y_columns = [col for col in data.columns if "sle" in col]

    # Concatenate all y-values to compute the overall upper and lower percentile thresholds
    all_y_values = pd.concat([data[col].dropna() for col in y_columns])
    upper_threshold = np.percentile(all_y_values, percentile)
    lower_threshold = np.percentile(all_y_values, 100 - percentile)

    # Iterate over each y-column to backfill outliers based on the overall upper and lower thresholds
    for col in y_columns:
        upper_extreme_value_mask = data[col] > upper_threshold
        lower_extreme_value_mask = data[col] < lower_threshold

        # Temporarily replace upper and lower extreme values with NaN
        data.loc[upper_extreme_value_mask, col] = np.nan
        data.loc[lower_extreme_value_mask, col] = np.nan

        # Use backfill method to fill NaN values
        data[col] = data[col].bfill()

    return data


def add_lag_variables(data: pd.DataFrame, lag: int, verbose=True) -> pd.DataFrame:
    """
    Adds lagged variables to the input dataset, creating time-shifted versions of the predictor variables.

    Args:
        data (pd.DataFrame): The dataset containing time series data.
        lag (int): The number of time steps to lag the variables.
        verbose (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        pd.DataFrame: The dataset with lagged variables added.
    """


    # Separate columns that won't be lagged and shouldn't be dropped
    cols_to_exclude = [x for x in data.columns if x not in ("year", "pr_anomaly", "evspsbl_anomaly", "mrro_anomaly", "smb_anomaly", "ts_anomaly", "thermal_forcing", "salinity", "temperature")]
    cols_to_exclude = [x for x in cols_to_exclude if x in data.columns]
    temporal_indicator = "time" if "time" in data.columns else "year"
    non_temporal_cols = [temporal_indicator] + [
        x for x in data.columns if "sle" in x or x in cols_to_exclude
    ]
    projection_length = 86

    # Initialize a list to collect the processed DataFrames
    processed_segments = []

    # Calculate the number of segments
    num_segments = len(data) // projection_length

    if verbose:
        iterator = tqdm(range(num_segments), total=num_segments, desc="Adding lag variables")
    else:
        iterator = range(num_segments)
    for segment_idx in iterator:
        # Extract the segment
        segment_start = segment_idx * projection_length
        segment_end = (segment_idx + 1) * projection_length
        segment = data.iloc[segment_start:segment_end, :]

        # Separate the segment into lagged and non-lagged parts
        non_lagged_data = segment[non_temporal_cols]
        base_temporal_columns = segment.drop(columns=non_temporal_cols)

        lags = []
        # Generate lagged variables for the segment
        for shift in range(1, lag + 1):
            lag_columns = base_temporal_columns.shift(shift).add_suffix(f".lag{shift}")
            # Fill missing values caused by shifting
            lag_columns.bfill(inplace=True)
            lags.append(lag_columns)
        full_segment_data = pd.concat([non_lagged_data.reset_index(drop=True), base_temporal_columns.reset_index(drop=True), pd.concat(lags, axis=1).reset_index(drop=True)], axis=1)


        # Store the processed segment
        processed_segments.append(full_segment_data)

    # Concatenate all processed segments into a single DataFrame
    final_data = pd.concat(processed_segments, axis=0)

    return final_data

def exclude_fetish_models(data: pd.DataFrame, exclude: str='both') -> pd.DataFrame:
    """
    Excludes specific models from the dataset.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    
    if exclude == '16km':
        return data[data.model != "fETISh_16km"]
    elif exclude == '32km':
        return data[data.model != "fETISh_32km"]
    elif exclude == 'both':
        return data[(data.model != "fETISh_16km") & (data.model != "fETISh_32km")]
    else:
        raise ValueError("exclude must be '16km', '32km', or 'both'")


def fill_mrro_nans(data: pd.DataFrame, method) -> pd.DataFrame:
    """
    Fills the NaN values in the specified columns with the given method.

    Args:
        data (pd.DataFrame): The input DataFrame.
        method (str or int): The method to fill NaN values. Must be one of 'zero', 'mean', 'median', or 'drop'.

    Returns:
        pd.DataFrame: The DataFrame with NaN values filled according to the specified method.

    Raises:
        ValueError: If the method is not one of 'zero', 'mean', 'median', or 'drop'.
    """
    mrro_columns = [x for x in data.columns if "mrro" in x]

    if method.lower() == "zero" or method.lower() == "0" or method == 0:
        for col in mrro_columns:
            data[col] = data[col].fillna(0)
    elif method.lower() == "mean":
        for col in mrro_columns:
            data[col] = data[col].fillna(data[col].mean())
    elif method.lower() == "median":
        for col in mrro_columns:
            data[col] = data[col].fillna(data[col].median())
    elif method.lower() == "drop":
        data = data.dropna(subset=mrro_columns)
    elif method.lower() == 'mean_by_year':
        data['mrro_anomaly'] = data.groupby('year')['mrro_anomaly'].transform(lambda x: x.fillna(x.mean()))
    else:
        raise ValueError("method must be 'zero', 'mean', 'median', or 'drop'")
    return data


def split_training_data(
    data, train_size, val_size, test_size=None, output_directory=None, random_state=42
):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        data (str or pd.DataFrame): The dataset or path to the dataset to be split.
        train_size (float): Proportion of data to use for training.
        val_size (float): Proportion of data to use for validation.
        test_size (float, optional): Proportion of data to use for testing. Defaults to the remainder.
        output_directory (str, optional): Directory to save the split datasets as CSV files. Defaults to None.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        tuple: Training, validation, and test datasets as pandas DataFrames.

    Raises:
        ValueError: If the dataset length is not divisible by 86, indicating incomplete projections.
        ValueError: If the dataset does not contain an 'id' column.
    """


    if isinstance(data, str):
        data = pd.read_csv(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a path (str) or a pandas DataFrame")

    if not len(data) % 86 == 0:
        warnings.warn(
            "Length of data must be divisible by 86, if not there are incomplete projections.")

    if "id" not in data.columns:
        raise ValueError("data must have a column named 'id'")

    total_ids = data["id"].unique()
    np.random.shuffle(total_ids)
    train_ids = total_ids[: int(len(total_ids) * train_size)]
    val_ids = total_ids[
        int(len(total_ids) * train_size) : int(len(total_ids) * (train_size + val_size))
    ]
    test_ids = total_ids[int(len(total_ids) * (train_size + val_size)) :]
    
    # train_ids = list(pd.read_csv(r'/oscar/scratch/pvankatw/datasets/sectors/GrIS/train.csv').id.unique())
    # val_ids = list(pd.read_csv(r'/oscar/scratch/pvankatw/datasets/sectors/GrIS/val.csv').id.unique())
    # test_ids = list(pd.read_csv(r'/oscar/scratch/pvankatw/datasets/sectors/GrIS/test.csv').id.unique())

    split_data = {
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
    }

    if output_directory is not None:
        with open(f"{output_directory}/ids.json", "w") as file:
            json.dump(split_data, file)

    train = data[data["id"].isin(train_ids)]
    val = data[data["id"].isin(val_ids)]
    test = data[data["id"].isin(test_ids)]

    if output_directory is not None:
        train.to_csv(f"{output_directory}/train.csv", index=False)
        val.to_csv(f"{output_directory}/val.csv", index=False)
        test.to_csv(f"{output_directory}/test.csv", index=False)

    return train, val, test


def drop_outliers(
    data: pd.DataFrame,
    column: str,
    method: str,
    expression: List[tuple] = None,
    quantiles: List[float] = [0.01, 0.99],
):
    """
    Removes outliers from the dataset based on a specified method.

    Args:
        data (pd.DataFrame): The dataset containing the column with potential outliers.
        column (str): The column to assess for outliers.
        method (str): The method of outlier detection ('quantile' or 'explicit').
        expression (list of tuples, optional): A list of conditions in the format [(column, operator, value)] for explicit filtering. Defaults to None.
        quantiles (list of float, optional): Quantiles for filtering when using the 'quantile' method. Defaults to [0.01, 0.99].

    Returns:
        pd.DataFrame: The dataset with outliers removed.

    Raises:
        AttributeError: If the method is 'quantile' but no quantiles are provided.
        AttributeError: If the method is 'explicit' but no expression is provided.
        ValueError: If the operator in the expression is not recognized.
    """


    # Check if method is quantile
    if method.lower() == "quantile":
        if quantiles is None:
            raise AttributeError("If method == quantile, quantiles argument cannot be None")

        # Calculate lower and upper quantiles
        lower_sle, upper_sle = np.quantile(np.array(data[column]), quantiles)

        # Filter outlier data based on quantiles
        outlier_data = data[(data[column] <= lower_sle) | (data[column] >= upper_sle)]

    # Check if method is explicit
    elif method.lower() == "explicit":
        if expression is None:
            raise AttributeError("If method == explicit, expression argument cannot be None")
        elif not isinstance(expression, list) or not isinstance(expression[0], tuple):
            raise AttributeError(
                'Expression argument must be a list of tuples, e.g. [("sle", ">", 20), ("sle", "<", -20)]'
            )

        outlier_data = data.copy()

        # Apply subset expressions to filter outlier data
        subset_dfs = []
        for subset_expression in expression:
            column, operator, value = subset_expression

            if operator.lower() in ("equal", "equals", "=", "=="):
                outlier_dataframe = outlier_data[outlier_data[column] == value]
            elif operator.lower() in ("not equal", "not equals", "!=", "~="):
                outlier_dataframe = outlier_data[outlier_data[column] != value]
            elif operator.lower() in ("greater than", "greater", ">=", ">"):
                outlier_dataframe = outlier_data[outlier_data[column] > value]
            elif operator.lower() in ("less than", "less", "<=", "<"):
                outlier_dataframe = outlier_data[outlier_data[column] < value]
            else:
                raise ValueError(f'Operator must be in ["==", "!=", ">", "<"], received {operator}')
            subset_dfs.append(outlier_dataframe)
        outlier_data = pd.concat(subset_dfs)

    # Check if outlier_data is empty
    if outlier_data.empty:
        return data

    # Create dataframe of experiments with outliers (want to delete the entire 86 rows)
    outlier_runs = pd.DataFrame()
    # TODO: Check to see if this works
    outlier_runs["modelname"] = outlier_data["model"]
    outlier_runs["exp_id"] = outlier_data["exp"]
    try:
        outlier_runs["sector"] = outlier_data["sector"]
        sectors = True
    except KeyError:
        sectors = False
    outlier_runs_list = outlier_runs.values.tolist()
    unique_outliers = [list(x) for x in set(tuple(x) for x in outlier_runs_list)]

    data["outlier"] = False

    # Drop those runs
    for i in tqdm(unique_outliers, total=len(unique_outliers), desc="Dropping outliers"):
        modelname = i[0]
        exp_id = i[1]

        if sectors:
            sector = i[2]
            data.loc[
                (data.model == modelname) & (data.exp == exp_id) & (data.sector == sector),
                "outlier",
            ] = True
        else:
            data.loc[(data.model == modelname) & (data.exp == exp_id), "outlier"] = True

    data = data[data["outlier"] == False]

    return data
