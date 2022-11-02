from multiprocessing.sharedctypes import Value
import pandas as pd
from sklearn import preprocessing as sp
from sklearn.model_selection import train_test_split
import numpy as np

class EmulatorData:
    def __init__(self, directory):
        
        self.directory = directory
        
        self.output_columns = ['icearea', 'iareafl', 'iareagr', 'ivol', 'ivaf', 'smb', 'smbgr', 'bmbfl']
        
        try:
            self.data = pd.read_csv(f"{self.directory}/master.csv", low_memory=False)
        except FileNotFoundError:
            try:
                self.inputs = pd.read_csv(f"{self.directory}/inputs.csv")
                self.outputs = pd.read_csv(f"{self.directory}/outputs.csv")
            except FileNotFoundError:
                raise FileNotFoundError('Files not found, make sure to run all processing functions.')
            
        self.X = None
        self.y = None
        self.scaler_X = None
        self.scaler_y = None
    
    def split_data(self, target_column: str, ):
        self.target_column = target_column
        self.X = self.data.drop(columns=self.output_columns)
        self.y = self.data[target_column]
        return self
    
    def train_test_split(self, train_size=0.7, shuffle=True):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=42, shuffle=False)
        return self
        
    def drop_missing(self):
        self.data = self.data.dropna()
        return self
    
    def create_boolean_indices(self, columns='all'):
        if columns == 'all':
            self.data = pd.get_dummies(self.data)
        else:
            if not isinstance(columns, list):
                raise ValueError(f'Columns argument must be of type: list, received {type(columns)}.')
            
            self.data = pd.get_dummies(self.data, columns=columns)
        return self
    
    def drop_columns(self, columns):
        if not isinstance(columns, list) and not isinstance(columns, str):
            raise ValueError(f'Columns argument must be of type: str|list, received {type(columns)}.')
        columns = list(columns)
        
        self.data = self.data.drop(columns=columns)
        return self
    
    def scale(self, values, values_type, scaler="MinMaxScaler"):
        if self.X is None and self.y is None:
            raise AttributeError('Data must be split before scaling using model.split_data method.')
        
        if "minmax" in scaler.lower():
            self.scaler_X = sp.MinMaxScaler()
            self.scaler_y = sp.MinMaxScaler()
        elif "standard" in scaler.lower():
            self.scaler_X = sp.StandardScaler()
            self.scaler_y = sp.StandardScaler()
        else:
            raise ValueError(f'scaler argument must be in [\'MinMaxScaler\', \'StandardScaler\'], received {scaler}')
        
        
        if 'input' in values_type.lower():
            self.input_columns = self.X.columns
            self.scaler_X.fit(self.X)
            return self.scaler_X.transform(values)
        
        elif 'output' in values_type.lower():
            self.scaler_y.fit(np.array(self.y).reshape(-1,1))
            return self.scaler_y.transform(np.array(values).reshape(-1,1))
        
        else:
            raise ValueError(f"values_type must be in ['inputs', 'outputs'], received {values_type}")

    
    def unscale(self, values, values_type):
        if not self.scaler_X and not self.scaler_y:
            raise AttributeError('Data has not been scaled.')
        
        if 'input' in values_type.lower():
            return pd.DataFrame(self.scaler_X.inverse_transform(values), columns=self.input_columns)
        
        elif 'output' in values_type.lower():
            return self.scaler_y.inverse_transform(values)
        
        else:
            raise ValueError(f"values_type must be in ['inputs', 'outputs'], received {values_type}")
        
    
        
            