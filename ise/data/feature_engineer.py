import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tqdm import tqdm

class FeatureEngineer:
    """
    A class for feature engineering operations on a given dataset.
    
    Args:
        data (pd.DataFrame): The input dataset.
        fill_mrro_nans (bool, optional): Flag indicating whether to fill missing values in the 'mrro' column. Defaults to False.
        split_dataset (bool, optional): Flag indicating whether to split the dataset into training, validation, and test sets. Defaults to False.
        train_size (float, optional): The proportion of the dataset to be used for training. Defaults to 0.7.
        val_size (float, optional): The proportion of the dataset to be used for validation. Defaults to 0.15.
        test_size (float, optional): The proportion of the dataset to be used for testing. Defaults to 0.15.
        output_directory (str, optional): The directory to save the split datasets. Defaults to None.
    """
    
    def __init__(self, ice_sheet, data: pd.DataFrame, fill_mrro_nans: bool=False, split_dataset: bool=False, train_size: float=0.7, val_size: float=0.15, test_size: float=0.15, output_directory: str=None):
        self.data = data
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.output_directory = output_directory
        

        self.scaler_X_path = None
        self.scaler_y_path = None
        self.scaler_X = None
        self.scaler_y = None
        
        if fill_mrro_nans:
            self.data = self.fill_mrro_nans(method='zero')
        
        if split_dataset:
            self.train, self.val, self.test = self.split_data(data, train_size, val_size, test_size, output_directory, random_state=42)
            
        
            
        self.train = None
        self.val = None
        self.test = None
    
    def split_data(self, data=None, train_size=None, val_size=None, test_size=None, output_directory=None, random_state=42):
        """
        Splits the dataset into training, validation, and test sets.
        
        Args:
            data (pd.DataFrame, optional): The input dataset. If not provided, the class attribute 'data' will be used. Defaults to None.
            train_size (float, optional): The proportion of the dataset to be used for training. If not provided, the class attribute 'train_size' will be used. Defaults to None.
            val_size (float, optional): The proportion of the dataset to be used for validation. If not provided, the class attribute 'val_size' will be used. Defaults to None.
            test_size (float, optional): The proportion of the dataset to be used for testing. If not provided, the class attribute 'test_size' will be used. Defaults to None.
            output_directory (str, optional): The directory to save the split datasets. If not provided, the class attribute 'output_directory' will be used. Defaults to None.
            random_state (int, optional): The random seed for reproducibility. Defaults to 42.
        
        Returns:
            tuple: A tuple containing the training, validation, and test sets.
        """
        if data is not None:
            self.data = data
        if train_size is not None:
            self.train_size = train_size
        if val_size is not None:
            self.val_size = val_size
        if output_directory is not None:
            self.output_directory = output_directory
            
        self.train, self.val, self.test = split_training_data(self.data, self.train_size, self.val_size, self.test_size, self.output_directory, random_state)
        return self.train, self.val, self.test
    
    def fill_mrro_nans(self, method, data=None):
        """
        Fills missing values in the 'mrro' column of the dataset.
        
        Args:
            method (str): The method to use for filling missing values.
            data (pd.DataFrame, optional): The input dataset. If not provided, the class attribute 'data' will be used. Defaults to None.
        
        Returns:
            pd.DataFrame: The dataset with missing values in the 'mrro' column filled.
        """
        if data is not None:
            self.data = data
        self.data = fill_mrro_nans(self.data, method)
    
        return self.data
    
    def scale_data(self, X=None, y=None, method='standard', save_dir=None):
        if X is not None:
            self.X = X
        else:
            self.X = self.data.drop(columns=[x for x in self.data.columns if 'sle' in x] + ['cmip_model', 'pathway', 'exp',])
        
        if y is not None:
            self.y = y 
        else:
            self.y = self.data[[x for x in self.data.columns if 'sle' in x]]

        if self.scaler_X_path is not None and self.scaler_y_path is not None:
            scaler_X = pickle.load(open(self.scaler_X_path, 'rb'))
            scaler_y = pickle.load(open(self.scaler_y_path, 'rb'))
            
            return scaler_X.transform(self.X), scaler_y.transform(self.y)
        elif self.scaler_X is not None and self.scaler_y is not None:
            return self.scaler_X.transform(self.X), self.scaler_y.transform(self.y)
        
        
        if (self.X is None and X is None) or (self.y is None and y is None):
            raise ValueError("X and y must be provided if they are not already stored in the class instance.")
        
        # Initialize the scalers based on the method
        if method == 'standard':
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif method == 'minmax':
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        elif method == 'robust':
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
        
        # Fit and transform X
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
            self.scaler_X_path = f"{save_dir}/scaler_X.pkl"
            self.scaler_y_path = f"{save_dir}/scaler_y.pkl"
            with open(self.scaler_X_path, 'wb') as f:
                pickle.dump(scaler_X, f)
            with open(self.scaler_y_path, 'wb') as f:
                pickle.dump(scaler_y, f)
        
        self.data = pd.concat([pd.DataFrame(X_scaled, columns=self.X.columns, index=self.X.index), pd.DataFrame(y_scaled, columns=self.y.columns, index=self.y.index)], axis=1)
                

        return X_scaled, y_scaled
    
    def unscale_data(self, X=None, y=None, scaler_X_path=None, scaler_y_path=None):
        
        if scaler_X_path is not None:
            self.scaler_X_path = scaler_X_path
        if scaler_y_path is not None:
            self.scaler_y_path = scaler_y_path
        
        # Load scaler for X
        if X is not None:
            if self.scaler_X_path is None:
                raise ValueError("scaler_X_path must be provided if X is not None.")
            with open(self.scaler_X_path, 'rb') as f:
                scaler_X = pickle.load(f)
            X_unscaled = scaler_X.inverse_transform(X)
            if isinstance(X, pd.DataFrame):
                X_unscaled = pd.DataFrame(X_unscaled, columns=X.columns, index=X.index)
        else:
            X_unscaled = None

        # Load scaler for y
        if y is not None:
            if self.scaler_y_path is None:
                raise ValueError("scaler_y_path must be provided if y is not None.")
            with open(self.scaler_y_path, 'rb') as f:
                scaler_y = pickle.load(f)
            y_unscaled = scaler_y.inverse_transform(y)
            if isinstance(y, pd.DataFrame):
                y_unscaled = pd.DataFrame(y_unscaled, columns=y.columns, index=y.index)
        else:
            y_unscaled = None

        return X_unscaled, y_unscaled
    
    def add_lag_variables(self, lag, data=None):
        if data is not None:
            self.data = data
        self.data = add_lag_variables(self.data, lag)
        return self
    
    def backfill_outliers(self, percentile=99.999, data=None):
        if data is not None:
            self.data = data
        self.data = backfill_outliers(self.data, percentile=percentile)
        return self
        
        
def backfill_outliers(data, percentile=99.999):
    """
    Replaces extreme values in y-values (above the specified percentile and below the 1-percentile across all y-values) 
    with the value from the previous row.

    Args:
        percentile (float): The percentile to use for defining upper extreme values across all y-values. Defaults to 99.999.
    """
    # Assuming y-values are in columns named with 'sle' as mentioned in other methods
    y_columns = [col for col in data.columns if 'sle' in col]
    
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
        data[col] = data[col].fillna(method='bfill')
        
    return data


    
def add_lag_variables(data: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    Adds lag variables to the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        lag (int): The number of time steps to lag the variables.

    Returns:
        pd.DataFrame: The DataFrame with lag variables added.
    """
    
    # Separate columns that won't be lagged and shouldn't be dropped
    cols_to_exclude = ['id', 'cmip_model', 'pathway', 'exp']
    non_lagged_cols = ['time'] + [x for x in data.columns if 'sle' in x or x in cols_to_exclude]
    projection_length = 86

    # Initialize a list to collect the processed DataFrames
    processed_segments = []

    # Calculate the number of segments
    num_segments = len(data) // projection_length

    for segment_idx in tqdm(range(num_segments), total=num_segments, desc='Adding lag variables'):
        # Extract the segment
        segment_start = segment_idx * projection_length
        segment_end = (segment_idx + 1) * projection_length
        segment = data.iloc[segment_start:segment_end, :]

        # Separate the segment into lagged and non-lagged parts
        non_lagged_data = segment[non_lagged_cols]
        lagged_data = segment.drop(columns=non_lagged_cols)
        
        # Generate lagged variables for the segment
        for shift in range(1, lag + 1):
            lagged_segment = lagged_data.shift(shift).add_suffix(f'.lag{shift}')
            # Fill missing values caused by shifting
            lagged_segment.fillna(method='bfill', inplace=True)
            non_lagged_data = pd.concat([non_lagged_data.reset_index(drop=True), lagged_segment.reset_index(drop=True)], axis=1)
        
        # Store the processed segment
        processed_segments.append(non_lagged_data)

    # Concatenate all processed segments into a single DataFrame
    final_data = pd.concat(processed_segments, axis=0)

    return final_data


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
    mrro_columns = [x for x in data.columns if 'mrro' in x]
    
    if method.lower() == 'zero' or method.lower() == '0' or method == 0:
        for col in mrro_columns:
            data[col] = data[col].fillna(0)
    elif method.lower() == 'mean':
        for col in mrro_columns:
            data[col] = data[col].fillna(data[col].mean())
    elif method.lower() == 'median':
        for col in mrro_columns:
            data[col] = data[col].fillna(data[col].median())
    elif method.lower() == 'drop':          
        data = data.dropna(subset=mrro_columns)
    else:
        raise ValueError("method must be 'zero', 'mean', 'median', or 'drop'")
    return data


def split_training_data(data, train_size, val_size, test_size=None, output_directory=None, random_state=42):
    """
    Splits the input data into training, validation, and test sets based on the specified sizes.

    Args:
        data (str or pandas.DataFrame): The input data to be split. It can be either a file path (str) or a pandas DataFrame.
        train_size (float): The proportion of data to be used for training.
        val_size (float): The proportion of data to be used for validation.
        test_size (float, optional): The proportion of data to be used for testing. If not provided, the remaining data after training and validation will be used for testing. Defaults to None.
        output_directory (str, optional): The directory where the split data will be saved as CSV files. Defaults to None.
        random_state (int, optional): The random seed for shuffling the data. Defaults to 42.

    Returns:
        tuple: A tuple containing the training, validation, and test sets as pandas DataFrames.

    Raises:
        ValueError: If the length of data is not divisible by 86, indicating incomplete projections.
        ValueError: If the data does not have a column named 'id'.

    """

    if isinstance(data, str):
        data = pd.read_csv(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a path (str) or a pandas DataFrame")
    
    if not len(data) % 86 == 0:
        raise ValueError("Length of data must be divisible by 86, if not there are incomplete projections.")
    
    if 'id' not in data.columns:
        raise ValueError("data must have a column named 'id'")
    
    total_ids = data['id'].unique()
    np.random.shuffle(total_ids)
    train_ids = total_ids[:int(len(total_ids)*train_size)]
    val_ids = total_ids[int(len(total_ids)*train_size):int(len(total_ids)*(train_size+val_size))]
    test_ids = total_ids[int(len(total_ids)*(train_size+val_size)):]
    
    train = data[data['id'].isin(train_ids)]
    val = data[data['id'].isin(val_ids)]
    test = data[data['id'].isin(test_ids)]
    
    if output_directory is not None:
        train.to_csv(f"{output_directory}/train.csv", index=False)
        val.to_csv(f"{output_directory}/val.csv", index=False)
        test.to_csv(f"{output_directory}/test.csv", index=False)

    
    return train, val, test