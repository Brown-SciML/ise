import numpy as np
import pandas as pd


import pandas as pd

import pandas as pd

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