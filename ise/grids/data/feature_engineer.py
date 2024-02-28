import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, data: pd.DataFrame, fill_mrro_nans: bool=False, split_dataset: bool=False, train_size: float=0.7, val_size: float=0.15, test_size: float=0.15, output_directory: str=None,):
        self.data = data
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.output_directory = output_directory
        
        if fill_mrro_nans:
            self.data = self.fill_mrro_nans(method='zero')
        
        if split_dataset:
            self.train, self.val, self.test = self.split_data(data, train_size, val_size, test_size, output_directory, random_state=42)
            
        self.trian = None
        self.val = None
        self.test = None
    
            
    def split_data(self, data=None, train_size=None, val_size=None, test_size=None, output_directory=None, random_state=42):
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
        if data is not None:
            self.data = data
        self.data = fill_mrro_nans(self.data, method)
    
        return self.data

def fill_mrro_nans(data: pd.DataFrame, method) -> pd.DataFrame:
    """
    Fill NaNs in the MRRO column with 0s
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
        data = data.dropna(subset=[mrro_columns])
    else:
        raise ValueError("method must be 'zero', 'mean', 'median', or 'drop'")
    return data


def split_training_data(data, train_size, val_size, test_size=None, output_directory=None, random_state=42):
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